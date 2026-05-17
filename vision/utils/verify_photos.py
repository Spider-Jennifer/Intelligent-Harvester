import os
from pathlib import Path

print("检查苹果照片...")
print("=" * 40)

# 检查照片目录
photo_dir = 'apple_photos'
if os.path.exists(photo_dir):
    photos = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
        photos.extend(list(Path(photo_dir).glob(f'*{ext}')))
    
    print(f'找到 {len(photos)} 张苹果照片')
    
    if len(photos) > 0:
        print('前10张照片:')
        for i, photo in enumerate(photos[:10]):
            print(f'  {i+1:2d}. {photo.name}')
    
    if len(photos) >= 50:
        print('状态: 照片数量足够，可以开始训练')
    elif len(photos) >= 20:
        print('状态: 照片数量基本足够，可以尝试训练')
    else:
        print('状态: 照片数量不足，建议至少50张')
else:
    print(f'错误: 找不到 {photo_dir} 目录')
    print('请将苹果照片放入该目录')

print("=" * 40)