import os
from pathlib import Path

def rename_images_in_directory():
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
    
    # 获取当前目录
    current_dir = Path('.')
    
    # 找到所有图片文件
    image_files = [
        f for f in current_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    # 按文件名排序
    image_files.sort(key=lambda x: x.name)
    
    if not image_files:
        print("当前目录没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件")
    print("\n将要进行以下重命名:")
    
    # 显示重命名计划
    for i, old_file in enumerate(image_files, 1):
        new_name = f"photo{i}.jpg"
        print(f"{old_file.name} -> {new_name}")
    
    # 确认是否继续
    confirm = input("\n确认重命名? (y/n): ")
    
    if confirm.lower() != 'y':
        print("已取消重命名")
        return
    
    # 执行重命名（先重命名为临时名称，避免冲突）
    temp_names = []
    for i, old_file in enumerate(image_files, 1):
        temp_name = current_dir / f"temp_{i}_{old_file.name}"
        old_file.rename(temp_name)
        temp_names.append(temp_name)
    
    # 重命名为最终名称
    for i, temp_file in enumerate(temp_names, 1):
        new_name = current_dir / f"photo{i}.jpg"
        temp_file.rename(new_name)
    
    print(f"\n成功重命名 {len(image_files)} 个文件！")

if __name__ == "__main__":
    rename_images_in_directory()