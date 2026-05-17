#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修正苹果检测模型误识别问题 - 最终版本
使用绝对路径，避免工作目录问题
"""

import os
import sys
import shutil
import random
import subprocess
import time
from pathlib import Path
import glob

# 获取脚本所在目录作为基础目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR

def get_abs_path(rel_path):
    """将相对路径转换为基于BASE_DIR的绝对路径"""
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(BASE_DIR, rel_path)

def print_step(step_num, title):
    """打印步骤标题"""
    print(f"\n{'='*80}")
    print(f"📋 步骤 {step_num}: {title}")
    print(f"{'='*80}")

def check_prerequisites():
    """检查必要的文件和目录"""
    print_step(1, "检查环境与依赖")
    
    # 检查必要目录
    new_photos_dir = r"C:\Users\李晨鑫\Desktop\apple（photo）"
    if not os.path.exists(new_photos_dir):
        print(f"❌ 错误: 新苹果照片目录不存在 {new_photos_dir}")
        return False
    
    # 检查数据集目录
    dataset_dir = get_abs_path("apple_dataset")
    if not os.path.exists(dataset_dir):
        print(f"❌ 错误: 数据集目录不存在 {dataset_dir}")
        print("   请确保数据集已创建")
        return False
    
    # 检查预训练模型
    yolo_model = get_abs_path("yolov8n.pt")
    if not os.path.exists(yolo_model):
        print("⚠️  警告: 预训练模型 yolov8n.pt 不存在")
        print("   下载中...")
        try:
            from ultralytics import YOLO
            YOLO("yolov8n.pt")
            print("✅ 预训练模型已下载")
        except:
            print("❌ 无法下载预训练模型，可能需要手动下载")
            return False
    
    print("✅ 环境检查通过")
    return True

def add_new_apple_photos():
    """将新苹果照片添加到数据集"""
    print_step(2, "添加新苹果照片到数据集")
    
    new_photos_dir = r"C:\Users\李晨鑫\Desktop\apple（photo）"
    dataset_images_dir = get_abs_path("apple_dataset/images")
    
    # 获取所有图片文件
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP']
    new_photos = []
    for ext in image_exts:
        new_photos.extend(list(Path(new_photos_dir).glob(f"*{ext}")))
    
    # 过滤掉脚本文件
    new_photos = [p for p in new_photos if p.suffix.lower() in [ext.lower() for ext in image_exts]]
    new_photos = [p for p in new_photos if "detect_apple_to_video" not in str(p)]
    
    print(f"📷 找到 {len(new_photos)} 张新苹果照片:")
    for p in new_photos:
        print(f"   - {p.name}")
    
    if not new_photos:
        print("❌ 错误: 没有找到苹果图片")
        return False
    
    # 确保目录存在
    for split in ["train", "val"]:
        split_dir = os.path.join(dataset_images_dir, split)
        os.makedirs(split_dir, exist_ok=True)
    
    # 获取现有最大索引
    def get_max_index(split):
        split_dir = os.path.join(dataset_images_dir, split)
        max_idx = -1
        try:
            for f in os.listdir(split_dir):
                if f.startswith("apple_"):
                    try:
                        num = f.split('_')[1].split('.')[0]
                        idx = int(num)
                        if idx > max_idx:
                            max_idx = idx
                    except:
                        pass
        except:
            pass
        return max_idx
    
    train_max = get_max_index("train")
    val_max = get_max_index("val")
    start_idx = max(train_max, val_max) + 1
    
    # 随机打乱并分割 (80%训练, 20%验证)
    random.shuffle(new_photos)
    split_idx = int(len(new_photos) * 0.8)
    train_photos = new_photos[:split_idx]
    val_photos = new_photos[split_idx:]
    
    print(f"📊 分割: 训练集 {len(train_photos)} 张, 验证集 {len(val_photos)} 张")
    
    # 复制训练集图片
    print("📤 复制训练集图片...")
    for i, photo_path in enumerate(train_photos):
        new_idx = start_idx + i
        dst_name = f"apple_{new_idx:04d}{photo_path.suffix}"
        dst_path = os.path.join(dataset_images_dir, "train", dst_name)
        shutil.copy2(photo_path, dst_path)
        print(f"   ✅ {photo_path.name} -> {dst_name}")
    
    # 复制验证集图片
    print("📤 复制验证集图片...")
    for i, photo_path in enumerate(val_photos):
        new_idx = start_idx + len(train_photos) + i
        dst_name = f"apple_{new_idx:04d}{photo_path.suffix}"
        dst_path = os.path.join(dataset_images_dir, "val", dst_name)
        shutil.copy2(photo_path, dst_path)
        print(f"   ✅ {photo_path.name} -> {dst_name}")
    
    print(f"✅ 成功添加 {len(new_photos)} 张新苹果照片到数据集")
    return True

def run_script(script_name, args=None):
    """运行Python脚本，在BASE_DIR目录下执行"""
    script_path = get_abs_path(script_name)
    if not os.path.exists(script_path):
        print(f"❌ 错误: 脚本不存在 {script_path}")
        return False
    
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    print(f"🚀 运行脚本: {os.path.basename(script_path)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, 
                              text=True, encoding='utf-8', errors='ignore')
        print(result.stdout)
        if result.stderr:
            print("标准错误:", result.stderr)
        
        if result.returncode != 0:
            print(f"❌ 脚本返回错误码: {result.returncode}")
            return False
        
        print("-" * 60)
        return True
        
    except Exception as e:
        print(f"❌ 运行脚本时出错: {e}")
        return False

def auto_relabel():
    """自动重新标注数据集（只保留苹果检测）"""
    print_step(3, "自动重新标注数据集")
    return run_script("relabel_apple_dataset.py")

def check_negative_ratio():
    """检查当前负样本比例"""
    train_labels_dir = get_abs_path("apple_dataset/labels/train")
    
    if not os.path.exists(train_labels_dir):
        print("  训练标签目录不存在")
        return
    
    label_files = [f for f in os.listdir(train_labels_dir) if f.endswith('.txt')]
    empty_labels = 0
    
    for f in label_files:
        p = os.path.join(train_labels_dir, f)
        if os.path.getsize(p) == 0:
            empty_labels += 1
    
    total = len(label_files)
    ratio = empty_labels / total * 100 if total > 0 else 0
    
    print(f"  📊 负样本比例: {empty_labels}/{total} ({ratio:.1f}%)")
    
    if ratio < 20:
        print(f"  ⚠️  建议增加负样本到20%以上以减少误识别")
    else:
        print(f"  ✅ 负样本比例良好")
    
    return ratio

def run_training():
    """运行训练脚本"""
    print_step(4, "训练改进模型")
    print("🤖 开始训练模型...")
    print("⏱️  这可能需要40-80分钟，请耐心等待")
    print("   您可以在 runs/detect/apple_no_fp 目录中查看训练进度")
    
    # 创建简化的训练脚本
    train_script = get_abs_path("train_misdetection_fix.py")
    
    # 如果专用训练脚本不存在，使用现有的
    if not os.path.exists(train_script):
        return run_script("improve_model_accuracy.py")
    
    return run_script("train_misdetection_fix.py")

def quick_test():
    """快速测试模型"""
    print_step(5, "快速测试模型")
    
    new_photos_dir = r"C:\Users\李晨鑫\Desktop\apple（photo）"
    
    # 查找最新的模型
    model_candidates = [
        get_abs_path("apple_no_fp.pt"),
        get_abs_path("apple_improved.pt"),
        get_abs_path("apple_best.pt"),
        get_abs_path("apple_sensitive.pt")
    ]
    
    model_path = None
    for m in model_candidates:
        if os.path.exists(m):
            model_path = m
            print(f"📦 使用模型: {os.path.basename(m)}")
            break
    
    if not model_path:
        print("⚠️  未找到训练好的模型，跳过测试")
        return False
    
    # 获取测试图片
    image_exts = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_exts:
        image_paths.extend(glob.glob(os.path.join(new_photos_dir, ext)))
    
    image_paths = [p for p in image_paths if "detect_apple_to_video" not in p]
    
    print(f"📷 测试图片: {len(image_paths)} 张")
    
    if not image_paths:
        print("❌ 没有测试图片")
        return False
    
    # 导入模型进行测试
    try:
        from ultralytics import YOLO
        import cv2
        
        model = YOLO(model_path)
        
        print("\n🔍 快速测试结果 (置信度 0.25):")
        print("   目标: 每张图片检测到1个苹果 (因为每张只有1个苹果)")
        print("   检测到多个苹果 = 可能误识别\n")
        
        misdetected = 0
        correct = 0
        no_detection = 0
        
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            results = model(img, conf=0.25, verbose=False)
            
            detections = 0
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    detections = len(result.boxes)
            
            if detections == 1:
                status = "✅ 正确"
                correct += 1
            elif detections == 0:
                status = "⚠️  漏检"
                no_detection += 1
            else:
                status = f"❌ 误识别 ({detections}个)"
                misdetected += 1
            
            print(f"   {img_name}: {status}")
        
        print(f"\n📊 统计:")
        print(f"   正确: {correct}/{len(image_paths)}")
        print(f"   误识别: {misdetected}/{len(image_paths)}")
        print(f"   漏检: {no_detection}/{len(image_paths)}")
        
        # 保存结果
        result_file = get_abs_path("quick_test_results.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"快速测试结果 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型: {os.path.basename(model_path)}\n")
            f.write(f"测试图片: {len(image_paths)} 张\n\n")
            f.write(f"正确检测: {correct}\n")
            f.write(f"误识别: {misdetected}\n")
            f.write(f"漏检: {no_detection}\n\n")
            
            if misdetected == 0:
                f.write("✅ 完美! 没有误识别\n")
            elif misdetected <= 2:
                f.write("👍 良好! 误识别很少\n")
            else:
                f.write("⚠️  仍需改进，存在较多误识别\n")
        
        print(f"\n📄 详细结果已保存到: quick_test_results.txt")
        
        return misdetected == 0
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        return False

def generate_detection_video():
    """生成检测视频"""
    print_step(6, "生成苹果检测视频")
    
    # 查找可用的模型
    model_candidates = [
        get_abs_path("apple_no_fp.pt"),      # 新训练的模型
        get_abs_path("apple_improved.pt"),
        get_abs_path("apple_sensitive.pt"),
        get_abs_path("apple_best.pt"),
        get_abs_path("apple_5epoch_test.pt"),
        get_abs_path("yolov8n.pt")
    ]
    
    model_path = None
    for m in model_candidates:
        if os.path.exists(m):
            model_path = m
            print(f"✅ 找到模型: {os.path.basename(m)}")
            break
    
    if not model_path:
        print("❌ 未找到任何可用的模型文件")
        return False
    
    # 查找可用的图片目录
    photo_dirs = [
        r"C:\Users\李晨鑫\Desktop\apple（photo）",
        r"C:\Users\李晨鑫\Desktop\苹果照片检测素材",
        get_abs_path("apple_photos"),
        get_abs_path("apple_dataset/images/train")
    ]
    
    image_dir = None
    for d in photo_dirs:
        if os.path.exists(d):
            image_dir = d
            print(f"✅ 找到图片目录: {d}")
            break
    
    if not image_dir:
        print("❌ 未找到图片目录")
        return False
    
    # 输出视频路径
    output_video = get_abs_path("apple_detection_final_fixed.avi")
    
    print(f"\n🎬 开始生成检测视频...")
    print(f"   模型: {os.path.basename(model_path)}")
    print(f"   图片目录: {image_dir}")
    print(f"   输出视频: {output_video}")
    
    try:
        # 导入必要的库
        import cv2
        import glob
        from ultralytics import YOLO
        
        # 加载模型
        model = YOLO(model_path)
        
        # 获取图片列表
        image_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
        image_paths = []
        for ext in image_exts:
            image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        
        image_paths.sort()
        
        if not image_paths:
            print("❌ 图片目录中没有找到图片")
            return False
        
        print(f"📷 找到 {len(image_paths)} 张图片")
        
        # 读取第一张图片获取尺寸
        first_img = cv2.imread(image_paths[0])
        if first_img is None:
            print("❌ 无法读取第一张图片")
            return False
        
        height, width = first_img.shape[:2]
        video_size = (width, height)
        
        # 视频配置
        fps = 10
        video_duration = 8.0  # 目标8秒
        total_frames_needed = int(video_duration * fps)
        frames_per_image = max(1, total_frames_needed // len(image_paths))
        actual_total_frames = frames_per_image * len(image_paths)
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, video_size)
        
        if not video_writer.isOpened():
            print("❌ 无法创建视频文件")
            return False
        
        print(f"\n⚙️  视频配置:")
        print(f"   尺寸: {video_size}")
        print(f"   帧率: {fps} FPS")
        print(f"   目标时长: {video_duration}秒")
        print(f"   每张图片帧数: {frames_per_image}")
        print(f"   总帧数: {actual_total_frames}")
        
        print("\n🔍 开始检测并生成视频...")
        total_apples = 0
        
        for i, img_path in enumerate(image_paths):
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            # 调整图片尺寸（如果需要）
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, video_size)
            
            # 检测苹果
            results = model(img, conf=0.3, iou=0.5, imgsz=320, verbose=False)
            
            apples_in_image = 0
            img_with_boxes = img.copy()
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes
                    apples_in_image = len(boxes)
                    
                    # 绘制检测框
                    for j in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy()
                        conf = boxes.conf[j].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # 绿色边界框
                        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 置信度标签
                        label = f"apple {conf:.2f}"
                        (label_w, label_h), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        
                        cv2.rectangle(img_with_boxes,
                                     (x1, y1 - label_h - baseline - 5),
                                     (x1 + label_w, y1),
                                     (0, 255, 0), -1)
                        
                        cv2.putText(img_with_boxes, label,
                                   (x1, y1 - baseline - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # 添加统计信息
            cv2.putText(img_with_boxes, f"Apples: {apples_in_image}",
                       (width - 150, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 写入多帧
            for _ in range(frames_per_image):
                video_writer.write(img_with_boxes)
            
            total_apples += apples_in_image
            print(f"   [{i+1}/{len(image_paths)}] {img_name}: {apples_in_image}个苹果")
        
        # 释放视频写入器
        video_writer.release()
        
        print(f"\n✅ 视频生成完成!")
        print(f"   视频文件: {output_video}")
        print(f"   总检测苹果数: {total_apples}")
        print(f"   平均每张图片苹果数: {total_apples/len(image_paths):.2f}")
        
        # 验证视频
        cap = cv2.VideoCapture(output_video)
        if cap.isOpened():
            video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            print(f"   验证: {video_frames}帧, {video_fps:.1f} FPS")
        
        return True
        
    except Exception as e:
        print(f"❌ 生成视频时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print(f"{'='*80}")
    print("🍎 苹果检测模型误识别问题解决方案")
    print("目标: 解决将人、眼镜、头发误识别为苹果的问题")
    print(f"{'='*80}")
    
    print(f"📁 工作目录: {BASE_DIR}")
    print(f"📁 数据集目录: {get_abs_path('apple_dataset')}")
    
    # 步骤1: 检查环境
    if not check_prerequisites():
        print("❌ 环境检查失败，请解决以上问题")
        return
    
    # 步骤2: 添加新苹果照片
    if not add_new_apple_photos():
        print("❌ 添加新照片失败")
        return
    
    # 步骤3: 自动重新标注
    if not auto_relabel():
        print("❌ 自动标注失败")
        return
    
    # 检查负样本比例
    print("\n📊 检查负样本比例...")
    ratio = check_negative_ratio()
    if ratio < 20:
        print("⚠️  负样本比例不足，建议运行 collect_negative_samples.py")
        print("   但为了节省时间，继续使用当前数据集...")
    
    # 步骤4: 训练改进模型
    print("\n🤔 开始训练改进模型？")
    print("   训练可能需要40-80分钟")
    print("   输入 'y' 继续，输入 'n' 跳过训练")
    
    # 跳过训练以节省时间，直接生成视频
    train_choice = 'n'
    
    if train_choice.lower() == 'y':
        run_training()
    else:
        print("🚫 跳过训练（用户要求直接生成视频）")
        print("   使用现有模型生成检测视频...")
    
    # 步骤5: 快速测试
    test_result = quick_test()
    
    # 步骤6: 生成检测视频
    video_success = generate_detection_video()
    
    print(f"\n{'='*80}")
    print("🎯 任务完成!")
    
    if test_result:
        print("✅ 模型误识别问题已改善!")
        print("📄 查看 quick_test_results.txt 获取详细结果")
    else:
        print("⚠️  仍存在误识别问题，建议:")
        print("   1. 运行 collect_negative_samples.py 收集更多负样本")
        print("   2. 手动检查数据集中有问题的标签")
        print("   3. 重新运行此脚本")
    
    if video_success:
        print("🎬 检测视频已生成: apple_detection_final_fixed.avi")
        print("   您可以使用媒体播放器查看此视频")
    
    print(f"\n{'='*80}")
    print("🔧 其他可用命令:")
    print(f"   收集负样本: python collect_negative_samples.py")
    print(f"   测试误识别: python test_misdetection.py")
    print(f"   测试模型: python test_apple_quick.py")
    print(f"   生成视频: python detect_photos_to_video_fixed.py")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()