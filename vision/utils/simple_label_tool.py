"""
简单的苹果标注工具
使用方法：
1. 将苹果图片放在 YOLO-v8-app/dataset/images/train/ 目录下
2. 运行此脚本：python simple_label_tool.py
3. 程序会自动为每张图片生成一个中心位置的苹果标注
"""

import os
import glob

def create_simple_labels():
    """为训练图片创建简单的中心位置标注"""
    image_dir = "YOLO-v8-app/dataset/images/train"
    label_dir = "YOLO-v8-app/dataset/labels/train"
    
    # 确保标签目录存在
    os.makedirs(label_dir, exist_ok=True)
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(image_dir, f"*{ext.upper()}")))
    
    print(f"找到 {len(image_files)} 张训练图片")
    
    # 为每张图片创建标签文件
    for img_path in image_files:
        # 生成对应的标签文件名
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)
        
        # 如果标签文件已存在，跳过
        if os.path.exists(label_path):
            continue
        
        # 创建简单的中心位置标注
        # 格式：class_id center_x center_y width height
        # 这里假设苹果在图片中心，占图片的40%
        label_content = "0 0.5 0.5 0.4 0.4\n"
        
        with open(label_path, 'w') as f:
            f.write(label_content)
        
        print(f"创建标签: {label_name}")
    
    print(f"\n已完成！为 {len(image_files)} 张图片创建了标签文件")
    print("注意：这只是临时解决方案，建议使用专业标注工具进行精确标注")

if __name__ == "__main__":
    create_simple_labels()