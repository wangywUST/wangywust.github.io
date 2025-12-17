import os
import io
from tqdm import tqdm  # 必须安装: pip install tqdm
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# 定义 MIME 类型常量
FOLDER_MIME = "application/vnd.google-apps.folder"

# ==========================================
# 核心组件：带速度监控的文件包装器
# ==========================================
class ProgressFile(io.IOBase):
    """
    包装文件对象，拦截 read() 调用以更新 tqdm 进度条，从而计算速度
    """
    def __init__(self, filepath, chunk_size=1024*1024):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.file_obj = open(filepath, 'rb')
        self.total_size = os.path.getsize(filepath)
        
        # 初始化进度条
        # unit='B': 单位是字节
        # unit_scale=True: 自动转换为 KB, MB, GB
        # unit_divisor=1024: 按照 1024 进制计算
        self.pbar = tqdm(
            total=self.total_size, 
            unit='B', 
            unit_scale=True, 
            unit_divisor=1024,
            desc=self.filename, 
            leave=True,
            # 自定义格式，让速度显示更显眼
            # {rate_fmt} 就是速度，例如 "12.5MB/s"
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

    def read(self, size=-1):
        # PyDrive2 读取数据时，会触发这里
        data = self.file_obj.read(size)
        if data:
            self.pbar.update(len(data))
        return data

    def close(self):
        self.pbar.close()
        self.file_obj.close()
    
    # 必要的辅助方法
    def tell(self): return self.file_obj.tell()
    def seek(self, offset, whence=0): return self.file_obj.seek(offset, whence)


# ==========================================
# 认证与文件夹逻辑 (保持不变)
# ==========================================
def auth_drive(client_secrets_path: str) -> GoogleDrive:
    gauth = GoogleAuth()
    gauth.settings['get_refresh_token'] = True 
    gauth.LoadClientConfigFile(client_secrets_path)
    gauth.LoadCredentialsFile("credentials.json")
    
    if gauth.credentials is None:
        try:
            gauth.LocalWebserverAuth(client_kwargs={"prompt": "consent"})
        except:
            gauth.CommandLineAuth()
    elif gauth.access_token_expired:
        try:
            gauth.Refresh()
        except:
            # 刷新失败时重新认证
            gauth.LocalWebserverAuth(client_kwargs={"prompt": "consent"})
    else:
        gauth.Authorize()
        
    gauth.SaveCredentialsFile("credentials.json")
    return GoogleDrive(gauth)

def get_folder_id(drive, parent_id, folder_name):
    q = f"'{parent_id}' in parents and mimeType='{FOLDER_MIME}' and trashed=false and title='{folder_name}'"
    file_list = drive.ListFile({'q': q}).GetList()
    if file_list:
        return file_list[0]['id']
    
    # print(f"Creating folder: {folder_name}") # 减少刷屏，注释掉
    metadata = {'title': folder_name, 'mimeType': FOLDER_MIME, 'parents': [{'id': parent_id}]}
    folder = drive.CreateFile(metadata)
    folder.Upload()
    return folder['id']

# ==========================================
# 上传逻辑
# ==========================================
def upload_folder_to_google_drive(local_folder, drive_path, client_secrets_path):
    drive = auth_drive(client_secrets_path)
    print(">>> 认证成功，准备开始上传...")

    path_parts = [p for p in drive_path.strip('/').split('/') if p]
    parent_id = 'root'
    for part in path_parts:
        parent_id = get_folder_id(drive, parent_id, part)

    print(f">>> 目标文件夹 ID: {parent_id}\n")

    for root, _, files in os.walk(local_folder):
        rel = os.path.relpath(root, local_folder)
        target_parent = parent_id
        if rel != '.':
            for seg in rel.split(os.sep):
                target_parent = get_folder_id(drive, target_parent, seg)
                
        for file_name in files:
            file_path = os.path.join(root, file_name)
            
            wrapped_file = None
            try:
                gfile = drive.CreateFile({
                    'title': file_name, 
                    'parents': [{'id': target_parent}]
                })
                
                # 使用包装器接管文件读取
                wrapped_file = ProgressFile(file_path)
                gfile.content = wrapped_file 
                
                # 开始上传
                gfile.Upload()
                
            except Exception as e:
                print(f"\nError uploading {file_name}: {e}")
            finally:
                if wrapped_file:
                    wrapped_file.close()

if __name__ == "__main__":
    client_secrets_path = './client_secret.json'
    local_path = './'
    drive_path = 'upload/test/'

    if not os.path.exists(local_path):
        print(f"Error: Local path '{local_path}' does not exist.")
    else:
        upload_folder_to_google_drive(local_path, drive_path, client_secrets_path)