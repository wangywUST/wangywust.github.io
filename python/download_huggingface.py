from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="wangywUCLA/firstmodel",
    repo_type="model",
    local_dir="downloaded_model"
)