from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="logs",
    repo_id="brjathu/HMR",
    repo_type="space",
)