import os
from pathlib import Path

# 检查照片目录
photo_dir = 'apple_photos'
if os.path.exists(photo_dir):
    photos = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
        photos.extend(list(Path(photo_dir).glob(f'*{ext}')))
    
    print(f'找到 {len(photos)} 张苹果照片')
    
    # 显示前5个文件名
    print('前5张照片:')
    for i, photo in enumerate(photos[:5]):
        print(f'  {i+1}. {photo.name}')
    
    if len(photos) >= 50:
        print('OK 照片数量足够，可以开始训练')
    else:
        print('警告: 照片数量不足，建议至少50张')
else:
    print('错误: 找不到 apple_photos 目录')