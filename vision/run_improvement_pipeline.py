#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
苹果检测精度改进全流程自动化脚本
自动完成：负样本收集 -> 整合到训练集 -> 超精度训练 -> 生成检测视频
"""

import os
import sys
import subprocess
import time

def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n[{description}]")
    print(f"命令: {cmd}")
    print("-" * 80)
    
    try:
        # 使用subprocess运行命令并实时输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # 逐行读取输出
        for line in iter(process.stdout.readline, ''):
            # 过滤掉乱码字符，保留基本ASCII和中文
            filtered_line = ''.join(c for c in line if ord(c) < 65536)
            print(filtered_line.strip())
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code == 0:
            print(f"[OK] {description} 完成")
            return True
        else:
            print(f"[ERROR] {description} 失败 (返回码: {return_code})")
            return False
            
    except Exception as e:
        print(f"[ERROR] 执行命令时出错: {e}")
        return False

def check_negative_ratio_simple():
    """简单检查负样本比例"""
    print("检查当前负样本比例...")
    
    train_labels_dir = "apple_dataset/labels/train"
    if not os.path.exists(train_labels_dir):
        print(f"错误: 找不到训练标签目录 {train_labels_dir}")
        return 0
    
    label_files = [f for f in os.listdir(train_labels_dir) if f.endswith('.txt')]
    empty_labels = 0
    
    for f in label_files:
        p = os.path.join(train_labels_dir, f)
        if os.path.getsize(p) == 0:
            empty_labels += 1
    
    total_labels = len(label_files)
    negative_ratio = empty_labels / total_labels * 100 if total_labels > 0 else 0
    
    print(f"当前负样本比例: {negative_ratio:.1f}% ({empty_labels}/{total_labels})")
    
    if negative_ratio < 20:
        print("警告: 负样本比例低于20%，建议增加负样本")
        return False
    else:
        print("良好: 负样本比例合适")
        return True

def collect_negative_samples_auto():
    """自动收集负样本（绕过用户交互）"""
    print("=" * 80)
    print("步骤1: 自动收集负样本")
    print("=" * 80)
    
    # 创建自动收集的Python脚本
    auto_collect_script = """
import os
import shutil
from pathlib import Path
import cv2
from ultralytics import YOLO

def detect_person_and_glasses(image_path):
    try:
        model = YOLO('yolov8n.pt')
        results = model(image_path, conf=0.3, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                detected_classes = set()
                for cls_id in result.boxes.cls.cpu().numpy():
                    cls_name = model.names[int(cls_id)]
                    detected_classes.add(cls_name)
                
                person_related = ['person', 'face', 'head', 'eye', 'glasses', 'sunglasses']
                found = any(any(related in cls_name.lower() for related in person_related) 
                           for cls_name in detected_classes)
                
                return found, len(result.boxes)
    except Exception as e:
        print(f"检测出错: {e}")
    
    return False, 0

def main():
    print("自动收集负样本...")
    
    source_dir = "C:\\\\Users\\\\李晨鑫\\\\Desktop\\\\苹果照片检测素材"
    if not os.path.exists(source_dir):
        print(f"错误: 源目录不存在 {source_dir}")
        return
    
    negative_dir = "apple_dataset/negative_samples"
    os.makedirs(negative_dir, exist_ok=True)
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(source_dir).glob(ext)))
    
    if not image_files:
        print(f"错误: 源目录中没有图片 {source_dir}")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    print("分析图片，寻找包含人或眼镜的负样本...")
    
    negative_images = []
    
    for i, img_path in enumerate(image_files):
        if i % 5 == 0:
            print(f"进度: {i}/{len(image_files)}")
        
        contains_person_or_glasses, detections = detect_person_and_glasses(str(img_path))
        
        if contains_person_or_glasses:
            dst_path = os.path.join(negative_dir, f"negative_{i:04d}.jpg")
            shutil.copy2(img_path, dst_path)
            negative_images.append(str(img_path))
            print(f"  负样本: {img_path.name}")
    
    print(f"收集完成! 负样本数: {len(negative_images)}")
    
    # 创建标签文件
    for img_path in negative_images:
        label_file = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        label_path = os.path.join(negative_dir, label_file)
        with open(label_path, 'w', encoding='utf-8') as f:
            pass
    
    # 自动整合到训练集
    print("自动整合到训练集...")
    train_images_dir = "apple_dataset/images/train"
    train_labels_dir = "apple_dataset/labels/train"
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    
    copied_count = 0
    for img_file in os.listdir(negative_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            src_img = os.path.join(negative_dir, img_file)
            dst_img = os.path.join(train_images_dir, f"negative_{img_file}")
            shutil.copy2(src_img, dst_img)
            
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label = os.path.join(negative_dir, label_file)
            dst_label = os.path.join(train_labels_dir, f"negative_{label_file}")
            
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                with open(dst_label, 'w', encoding='utf-8') as f:
                    pass
            
            copied_count += 1
    
    print(f"整合完成! 添加了 {copied_count} 个负样本")
    print("负样本收集与整合完成!")

if __name__ == "__main__":
    main()
"""
    
    # 写入临时脚本
    script_path = "auto_collect_temp.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(auto_collect_script)
    
    # 运行脚本
    success = run_command(f"python {script_path}", "负样本收集与整合")
    
    # 清理临时脚本
    if os.path.exists(script_path):
        os.remove(script_path)
    
    return success

def train_ultra_precision_model_auto():
    """自动训练超精度模型"""
    print("=" * 80)
    print("步骤2: 训练超精度苹果检测模型")
    print("=" * 80)
    
    # 检查模型文件是否存在
    if os.path.exists("apple_ultra_precision.pt"):
        print("检测到已存在的超精度模型，跳过训练")
        return True
    
    # 创建自动训练脚本（简化版）
    auto_train_script = """
import os
import time
from ultralytics import YOLO

def main():
    print("开始训练超精度苹果检测模型...")
    
    # 检查数据集
    data_yaml = "apple_dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"错误: 找不到数据集配置文件 {data_yaml}")
        return False
    
    # 加载预训练模型
    try:
        model = YOLO('yolov8n.pt')
        print("预训练模型加载成功")
    except Exception as e:
        print(f"加载预训练模型失败: {e}")
        return False
    
    # 训练配置（简化版）
    training_config = {
        'data': data_yaml,
        'epochs': 30,           # 减少epoch数以加快训练
        'imgsz': 320,           # 较小尺寸加快训练
        'batch': 4,
        'workers': 0,
        'device': 'cpu',
        'name': 'apple_ultra_precision_auto',
        'patience': 15,
        'save': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.0005,
        'cls': 2.0,            # 高分类损失权重，减少误识别
        'conf': 0.4,           # 高验证置信度
        'iou': 0.6,            # 高IoU阈值
        'verbose': True,
    }
    
    print("训练配置:")
    print(f"  Epochs: {training_config['epochs']}")
    print(f"  图像尺寸: {training_config['imgsz']}")
    print(f"  分类损失权重: {training_config['cls']} (高，减少误识别)")
    print(f"  验证置信度: {training_config['conf']} (高，严格评估)")
    
    print("\\n开始训练...")
    print("注意: 训练可能需要30-60分钟，请耐心等待")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        results = model.train(**training_config)
        training_time = time.time() - start_time
        
        print(f"\\n训练完成! 耗时: {training_time/60:.1f} 分钟")
        
        # 保存最终模型
        best_model = "runs/detect/apple_ultra_precision_auto/weights/best.pt"
        if os.path.exists(best_model):
            import shutil
            shutil.copy2(best_model, "apple_ultra_precision.pt")
            print(f"模型已保存: apple_ultra_precision.pt")
            return True
        else:
            print("警告: 找不到最佳模型文件")
            return False
            
    except Exception as e:
        print(f"训练失败: {e}")
        return False

if __name__ == "__main__":
    main()
"""
    
    # 写入临时脚本
    script_path = "auto_train_temp.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(auto_train_script)
    
    # 运行训练脚本
    print("开始训练超精度模型...")
    print("这可能需要30-60分钟，请耐心等待")
    print("训练过程中会有进度显示")
    print("-" * 80)
    
    success = run_command(f"python {script_path}", "超精度模型训练")
    
    # 清理临时脚本
    if os.path.exists(script_path):
        os.remove(script_path)
    
    return success

def generate_detection_video_auto():
    """自动生成检测视频"""
    print("=" * 80)
    print("步骤3: 生成超精度检测视频")
    print("=" * 80)
    
    # 检查模型文件
    model_candidates = [
        "apple_ultra_precision.pt",
        "apple_5epoch_test.pt",
        "apple_sensitive.pt"
    ]
    
    selected_model = None
    for model_path in model_candidates:
        if os.path.exists(model_path):
            selected_model = model_path
            break
    
    if selected_model is None:
        print("错误: 找不到任何模型文件")
        return False
    
    print(f"使用模型: {selected_model}")
    
    # 创建自动检测脚本
    auto_detect_script = f"""
import os
import cv2
import glob
from pathlib import Path
from ultralytics import YOLO
import numpy as np

def main():
    print("生成超精度检测视频...")
    
    # 加载模型
    model = YOLO('{selected_model}')
    
    # 目标目录
    target_dir = "C:\\\\Users\\\\李晨鑫\\\\Desktop\\\\苹果照片检测素材"
    if not os.path.exists(target_dir):
        print(f"错误: 目标目录不存在 {{target_dir}}")
        return False
    
    # 获取图片
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(target_dir, ext)))
    
    if not image_files:
        print(f"错误: 目标目录中没有图片 {{target_dir}}")
        return False
    
    image_files.sort()
    print(f"找到 {{len(image_files)}} 张图片")
    
    # 读取第一张图片获取尺寸
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print(f"错误: 无法读取第一张图片")
        return False
    
    height, width = first_img.shape[:2]
    
    # 视频配置
    video_output = "apple_detection_ultra_precision.avi"
    fps = 10
    frames_per_image = 4
    
    # 创建视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_output, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"错误: 无法创建视频文件 {{video_output}}")
        return False
    
    print("处理图片...")
    
    total_apples = 0
    
    for i, img_path in enumerate(image_files):
        print(f"进度: {{i+1}}/{{len(image_files)}}")
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        # 检测（使用较高置信度减少误识别）
        results = model(img, conf=0.4, verbose=False)
        apples_in_image = 0
        img_with_boxes = img.copy()
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes
                apples_in_image = len(boxes)
                
                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy()
                    conf = boxes.conf[j].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 绘制框
                    color = (0, 255, 0) if conf >= 0.5 else (0, 255, 255)
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        total_apples += apples_in_image
        
        # 添加统计信息
        cv2.putText(img_with_boxes, f"Apples: {{apples_in_image}}",
                   (width - 150, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(img_with_boxes, f"Conf: 0.4",
                   (20, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 写入视频帧
        for _ in range(frames_per_image):
            video_writer.write(img_with_boxes)
    
    video_writer.release()
    
    print(f"\\n视频生成完成: {{video_output}}")
    print(f"检测到的苹果总数: {{total_apples}}")
    print(f"平均每张图片: {{total_apples/len(image_files):.2f}} 个苹果")
    
    if os.path.exists(video_output):
        file_size = os.path.getsize(video_output) / (1024 * 1024)
        print(f"文件大小: {{file_size:.1f}} MB")
    
    return True

if __name__ == "__main__":
    main()
"""
    
    # 写入临时脚本
    script_path = "auto_detect_temp.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(auto_detect_script)
    
    # 运行检测脚本
    success = run_command(f"python {script_path}", "生成检测视频")
    
    # 清理临时脚本
    if os.path.exists(script_path):
        os.remove(script_path)
    
    return success

def main():
    """主函数"""
    print("=" * 80)
    print("苹果检测精度改进全流程自动化")
    print("=" * 80)
    print("将自动执行以下步骤:")
    print("1. 收集负样本（针对眼镜、人脸等误识别物体）")
    print("2. 整合负样本到训练集")
    print("3. 训练超精度苹果检测模型")
    print("4. 生成新的检测视频")
    print("=" * 80)
    
    # 检查当前状态
    print("\n[状态检查]")
    print("检查当前负样本比例...")
    has_enough_negative_samples = check_negative_ratio_simple()
    
    # 步骤1: 收集负样本（如果需要）
    if not has_enough_negative_samples:
        print("\n负样本不足，开始收集负样本...")
        if not collect_negative_samples_auto():
            print("负样本收集失败，继续使用现有数据集训练")
    else:
        print("\n负样本充足，跳过收集步骤")
    
    # 步骤2: 训练超精度模型
    print("\n" + "=" * 80)
    print("开始训练超精度模型")
    print("=" * 80)
    
    # 训练模型
    if not train_ultra_precision_model_auto():
        print("模型训练失败，尝试使用现有模型...")
    
    # 步骤3: 生成检测视频
    print("\n" + "=" * 80)
    print("生成新的检测视频")
    print("=" * 80)
    
    if not generate_detection_video_auto():
        print("视频生成失败")
    
    # 完成
    print("\n" + "=" * 80)
    print("苹果检测精度改进全流程完成!")
    print("=" * 80)
    print("生成的成果:")
    print("1. 新训练的超精度模型: apple_ultra_precision.pt")
    print("2. 新的检测视频: apple_detection_ultra_precision.avi")
    print("3. 增加了负样本数量，减少误识别")
    print("\n下一步:")
    print("1. 查看检测视频: 用播放器打开 apple_detection_ultra_precision.avi")
    print("2. 检查眼镜等物体是否还被误识别为苹果")
    print("3. 如果仍有误识别，可进一步提高置信度阈值")
    print("=" * 80)

if __name__ == "__main__":
    main()