import os
import glob

# 检查标签文件
label_dir = 'YOLO-v8-app/dataset/labels/train'
if os.path.exists(label_dir):
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    print(f'训练标签文件数量: {len(label_files)}')
    
    # 检查标签内容
    if label_files:
        for i, label_file in enumerate(label_files[:3]):  # 检查前3个文件
            sample_file = os.path.join(label_dir, label_file)
            print(f'\n检查文件 {i+1}: {sample_file}')
            try:
                with open(sample_file, 'r') as f:
                    content = f.read().strip()
                    print(f'内容: {content}')
                    if content:
                        lines = content.split('\n')
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                print(f'  类别: {parts[0]}, 坐标: {parts[1:5]}')
            except Exception as e:
                print(f'读取错误: {e}')
else:
    print(f'标签目录不存在: {label_dir}')

# 检查验证集标签
val_label_dir = 'YOLO-v8-app/dataset/labels/val'
if os.path.exists(val_label_dir):
    val_label_files = [f for f in os.listdir(val_label_dir) if f.endswith('.txt')]
    print(f'\n验证集标签文件数量: {len(val_label_files)}')
else:
    print(f'\n验证集标签目录不存在: {val_label_dir}')