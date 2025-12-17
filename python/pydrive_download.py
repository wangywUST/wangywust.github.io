"""
gdrive_download_by_name_fixed.py

用法：
- 在代码里修改 FOLDER_NAME 和 OUTPUT_DIR
- 运行 python gdrive_download_by_name_fixed.py
"""

import os
import time
import io
from tqdm import tqdm

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from googleapiclient.http import MediaIoBaseDownload



# =========================
# 👉 在这里改即可
# =========================
FOLDER_NAME = "test"          # Google Drive 里的文件夹名字
OUTPUT_DIR = "./downloaded_data"   # 本地保存路径
PARENT_FOLDER_ID = None       # 如果知道父目录ID，可填写；否则保持 None


# =========================
# 认证
# =========================
def authenticate():
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("credentials.json")

    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()

    gauth.SaveCredentialsFile("credentials.json")
    return GoogleDrive(gauth)


# =========================
# 按名字查找文件夹
# =========================
def find_folders_by_name(drive, folder_name, parent_id=None):
    q = (
        f"title='{folder_name}' and "
        f"mimeType='application/vnd.google-apps.folder' and "
        f"trashed=false"
    )
    if parent_id:
        q += f" and '{parent_id}' in parents"

    return drive.ListFile({'q': q}).GetList()


def choose_folder_id(folders):
    if not folders:
        raise RuntimeError("❌ 没有找到该名字的文件夹，请确认名称和权限")

    if len(folders) == 1:
        f = folders[0]
        print(f"✅ 找到文件夹: {f['title']} (id={f['id']})")
        return f['id']

    print("⚠️ 找到多个同名文件夹，请选择：")
    for i, f in enumerate(folders, 1):
        print(f"  [{i}] title='{f['title']}', id={f['id']}")

    while True:
        idx = input("输入序号选择：").strip()
        if idx.isdigit() and 1 <= int(idx) <= len(folders):
            f = folders[int(idx) - 1]
            print(f"✅ 选择: {f['title']} (id={f['id']})")
            return f['id']
        print("输入无效，请重试")


# =========================
# 带进度条下载文件
# =========================
def download_file_with_progress(file, output_path):
    """
    使用 Google 官方 MediaIoBaseDownload
    - 稳定
    - 有真实进度
    - 可计算速度
    """

    # 取 service（这是 PyDrive2 已经创建好的）
    service = file.auth.service

    request = service.files().get_media(fileId=file['id'])

    fh = io.FileIO(output_path, mode='wb')
    downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024)

    total_size = int(file.get('fileSize', 0))
    title = file.get('title', os.path.basename(output_path))

    pbar = tqdm(
        total=total_size if total_size > 0 else None,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        desc=title,
        leave=True
    )

    start_time = time.time()
    downloaded = 0

    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            downloaded = int(status.resumable_progress)
            elapsed = time.time() - start_time
            speed = downloaded / elapsed if elapsed > 0 else 0

            pbar.n = downloaded
            pbar.set_postfix(speed=f"{speed/1024/1024:.2f} MB/s")
            pbar.refresh()

    pbar.close()
    fh.close()



# =========================
# 递归下载文件夹
# =========================
def download_folder(drive, folder_id, local_path):
    os.makedirs(local_path, exist_ok=True)

    items = drive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=false"
    }).GetList()

    for item in items:
        title = item['title']
        mime = item['mimeType']
        fid = item['id']

        if mime == 'application/vnd.google-apps.folder':
            print(f"\n📁 进入文件夹: {title}")
            download_folder(drive, fid, os.path.join(local_path, title))
        else:
            out = os.path.join(local_path, title)
            print(f"\n⬇ 下载: {title}")
            download_file_with_progress(item, out)


# =========================
# 主流程
# =========================
if __name__ == "__main__":
    drive = authenticate()

    folders = find_folders_by_name(drive, FOLDER_NAME, PARENT_FOLDER_ID)
    folder_id = choose_folder_id(folders)

    print(f"\n🚀 开始下载文件夹 '{FOLDER_NAME}' 到 {OUTPUT_DIR}\n")
    download_folder(drive, folder_id, OUTPUT_DIR)

    print("\n✅ 全部下载完成")
