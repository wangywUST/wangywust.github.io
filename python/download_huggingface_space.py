from huggingface_hub import snapshot_download

# 在这里填入模型 ID，例如 "facebook/opt-125m"
repo_id = "thinkwee/DDR_Bench" 

# 指定下载到的本地路径，不填则默认下载到 ~/.cache/huggingface
local_dir = "./"

print(f"正在开始下载模型: {repo_id}...")

snapshot_download(
    repo_id=repo_id,
    repo_type="space",
    local_dir=local_dir,
    local_dir_use_symlinks=False, # Windows 建议设为 False 以避免权限问题
    resume_download=True          # 开启断点续传
)

print("下载完成！")