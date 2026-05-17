import os

# 创建数据集目录结构
dirs = [
    'YOLO-v8-app/dataset/images/train',
    'YOLO-v8-app/dataset/images/val',
    'YOLO-v8-app/dataset/labels/train',
    'YOLO-v8-app/dataset/labels/val'
]

for dir_path in dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f'创建目录: {dir_path}')

print('数据集目录结构已创建')