import argparse
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
import trimesh
import tempfile

from hmr2.configs import get_config
from hmr2.datasets.vitdet_dataset import (DEFAULT_MEAN, DEFAULT_STD,
                                          ViTDetDataset)
from hmr2.models import HMR2
from hmr2.utils import recursive_to
from hmr2.utils.renderer import Renderer, cam_crop_to_full

# Setup HMR2.0 model
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
DEFAULT_CHECKPOINT='logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_cfg = str(Path(DEFAULT_CHECKPOINT).parent.parent / 'model_config.yaml')
model_cfg = get_config(model_cfg)
model = HMR2.load_from_checkpoint(DEFAULT_CHECKPOINT, strict=False, cfg=model_cfg).to(device)
model.eval()


# Load detector
from detectron2.config import LazyConfig

from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy

detectron2_cfg = LazyConfig.load(f"vendor/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_h_75ep.py")
detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
for i in range(3):
    detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
detector = DefaultPredictor_Lazy(detectron2_cfg)

# Setup the renderer
renderer = Renderer(model_cfg, faces=model.smpl.faces)


import numpy as np


def infer(in_pil_img, in_threshold=0.8):

    open_cv_image = np.array(in_pil_img)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    print("EEEEE", open_cv_image.shape)
    det_out = detector(open_cv_image)
    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > in_threshold)
    boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

    # Run HMR2.0 on all detected humans
    dataset = ViTDetDataset(model_cfg, open_cv_image, boxes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    all_verts = []
    all_cam_t = []

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        pred_cam = out['pred_cam']
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        render_size = img_size
        pred_cam_t = cam_crop_to_full(pred_cam, box_center, box_size, render_size, focal_length=img_size.mean()*2).detach().cpu().numpy()

        # Render the result
        batch_size = batch['img'].shape[0]
        for n in range(batch_size):
            # Get filename from path img_path
            # img_fn, _ = os.path.splitext(os.path.basename(img_path))
            person_id = int(batch['personid'][n])
            white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
            input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
            input_patch = input_patch.permute(1,2,0).numpy()

            regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                    out['pred_cam_t'][n].detach().cpu().numpy(),
                                    batch['img'][n],
                                    mesh_base_color=LIGHT_BLUE,
                                    scene_bg_color=(1, 1, 1),
                                    )


            verts = out['pred_vertices'][n].detach().cpu().numpy()
            cam_t = pred_cam_t[n]

            all_verts.append(verts)
            all_cam_t.append(cam_t)

    # Return mesh path
    trimeshes = [renderer.vertices_to_trimesh(vvv, ttt.copy(), LIGHT_BLUE) for vvv,ttt in zip(all_verts, all_cam_t)]

    # Join meshes
    mesh = trimesh.util.concatenate(trimeshes)

    # Save mesh to file
    temp_name = next(tempfile._get_candidate_names()) + '.obj'
    trimesh.exchange.export.export_mesh(mesh, temp_name)
    return temp_name


with gr.Blocks(title="4DHumans", css=".gradio-container") as demo:

    gr.HTML("""<div style="font-weight:bold; text-align:center; color:royalblue;">HMR 2.0</div>""")

    with gr.Row():
        input_image = gr.Image(label="Input image", type="pil", width=300, height=300, fixed_size=True)
        output_model = gr.Model3D(label="Reconstructions", width=300, height=300, fixed_size=True, clear_color=[0.0, 0.0, 0.0, 0.0])

    gr.HTML("""<br/>""")

    with gr.Row():
        threshold = gr.Slider(0, 1.0, value=0.8, label='Detection Threshold')
        send_btn = gr.Button("Infer")
        send_btn.click(fn=infer, inputs=[input_image, threshold], outputs=[output_model])

    # gr.Examples(['samples/img1.jpg', 'samples/img2.png', 'samples/img3.jpg', 'samples/img4.jpg'], inputs=input_image)

    gr.HTML("""</ul>""")



#demo.queue()
demo.launch(debug=True)




### EOF ###
