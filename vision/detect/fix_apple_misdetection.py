#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修正苹果检测模型误识别问题 - 完整解决方案
专门解决将人、眼镜、头发误识别为苹果的问题
执行步骤：
1. 添加新苹果照片到数据集
2. 自动重新标注（只保留苹果检测）
3. 收集并添加负样本（减少误识别）
4. 训练改进模型（专门针对区分苹果和非苹果）
5. 测试模型并检查误识别
6. 根据结果迭代优化
"""

import os
import sys
import shutil
import random
import subprocess
import time
from pathlib import Path
import cv2
import glob

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
    dataset_dir = "apple_dataset"
    if not os.path.exists(dataset_dir):
        print(f"❌ 错误: 数据集目录不存在 {dataset_dir}")
        print("   请确保数据集已创建")
        return False
    
    # 检查预训练模型
    if not os.path.exists("yolov8n.pt"):
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
    
    # 获取所有图片文件
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP']
    new_photos = []
    for ext in image_exts:
        new_photos.extend(list(Path(new_photos_dir).glob(f"*{ext}")))
    
    # 过滤掉脚本文件
    new_photos = [p for p in new_photos if p.suffix.lower() in [ext.lower() for ext in image_exts]]
    
    # 排除detect_apple_to_video_8sec.py
    new_photos = [p for p in new_photos if "detect_apple_to_video" not in str(p)]
    
    print(f"📷 找到 {len(new_photos)} 张新苹果照片:")
    for p in new_photos:
        print(f"   - {p.name}")
    
    if not new_photos:
        print("❌ 错误: 没有找到苹果图片")
        return False
    
    # 数据集目录
    dataset_images_dir = "apple_dataset/images"
    
    # 确保目录存在
    for split in ["train", "val"]:
        os.makedirs(os.path.join(dataset_images_dir, split), exist_ok=True)
    
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

def auto_relabel():
    """自动重新标注数据集（只保留苹果检测）"""
    print_step(3, "自动重新标注数据集")
    
    # 检查relabel脚本是否存在
    if not os.path.exists("relabel_apple_dataset.py"):
        print("❌ 错误: relabel_apple_dataset.py 不存在")
        return False
    
    print("🔍 运行自动标注脚本...")
    print("-" * 60)
    
    try:
        result = subprocess.run([sys.executable, "relabel_apple_dataset.py"], 
                              capture_output=True, text=True, encoding='utf-8', errors='ignore')
        print(result.stdout)
        if result.stderr:
            print("标准错误:", result.stderr)
        
        if result.returncode != 0:
            print(f"❌ 自动标注失败，返回码: {result.returncode}")
            return False
        
        print("-" * 60)
        print("✅ 自动标注完成！")
        
        # 检查标注结果
        train_labels_dir = "apple_dataset/labels/train"
        if os.path.exists(train_labels_dir):
            label_files = [f for f in os.listdir(train_labels_dir) if f.endswith('.txt')]
            non_empty = 0
            for f in label_files[:10]:  # 检查前10个文件
                p = os.path.join(train_labels_dir, f)
                if os.path.getsize(p) > 0:
                    non_empty += 1
            
            print(f"📊 标签统计: {len(label_files)} 个标签文件，前10个中 {non_empty} 个非空")
        
        return True
        
    except Exception as e:
        print(f"❌ 运行自动标注脚本时出错: {e}")
        return False

def add_negative_samples():
    """添加负样本以减少误识别"""
    print_step(4, "添加负样本以减少误识别")
    
    # 检查是否有现有的负样本
    negative_samples_dir = "apple_dataset/negative_samples"
    
    if os.path.exists(negative_samples_dir):
        print("📁 发现现有负样本目录")
        negative_images = [f for f in os.listdir(negative_samples_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if negative_images:
            print(f"📸 发现 {len(negative_images)} 个负样本图片")
            
            # 检查是否已整合到训练集
            train_images_dir = "apple_dataset/images/train"
            integrated_count = 0
            for img_file in negative_images:
                negative_name = f"negative_{img_file}"
                if os.path.exists(os.path.join(train_images_dir, negative_name)):
                    integrated_count += 1
            
            print(f"📊 负样本已整合: {integrated_count}/{len(negative_images)}")
            
            # 如果没有完全整合，进行整合
            if integrated_count < len(negative_images):
                print("🔧 整合负样本到训练集...")
                from collect_negative_samples import integrate_with_training_dataset
                try:
                    count = integrate_with_training_dataset(negative_samples_dir)
                    print(f"✅ 成功整合 {count} 个负样本到训练集")
                except:
                    print("⚠️  整合失败，手动创建空标签")
                    # 手动创建空标签
                    self_create_negative_labels(negative_samples_dir)
            else:
                print("✅ 负样本已完全整合")
        else:
            print("⚠️  负样本目录中没有图片")
    else:
        print("⚠️  负样本目录不存在，创建新目录...")
        os.makedirs(negative_samples_dir, exist_ok=True)
    
    # 检查负样本比例
    print("\n📈 检查负样本比例...")
    check_negative_ratio()
    
    return True

def self_create_negative_labels(negative_dir):
    """手动为负样本创建空标签文件"""
    print("  手动创建负样本标签...")
    
    train_images_dir = "apple_dataset/images/train"
    train_labels_dir = "apple_dataset/labels/train"
    
    # 确保目录存在
    os.makedirs(train_labels_dir, exist_ok=True)
    
    # 复制负样本图片到训练集
    negative_images = [f for f in os.listdir(negative_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in negative_images:
        src_img = os.path.join(negative_dir, img_file)
        dst_img = os.path.join(train_images_dir, f"negative_{img_file}")
        
        # 复制图片
        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)
        
        # 创建对应的空标签文件
        label_file = os.path.splitext(f"negative_{img_file}")[0] + ".txt"
        dst_label = os.path.join(train_labels_dir, label_file)
        
        if not os.path.exists(dst_label):
            with open(dst_label, 'w', encoding='utf-8') as f:
                pass  # 空文件表示没有苹果
    
    print(f"  创建了 {len(negative_images)} 个负样本的标签")

def check_negative_ratio():
    """检查当前负样本比例"""
    train_labels_dir = "apple_dataset/labels/train"
    
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
        print(f"  可以运行: python collect_negative_samples.py")
    else:
        print(f"  ✅ 负样本比例良好")

def train_misdetection_focused_model():
    """训练专门针对减少误识别的模型"""
    print_step(5, "训练防误识别改进模型")
    
    # 检查数据集配置
    data_yaml = "apple_dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"❌ 错误: 数据集配置文件不存在 {data_yaml}")
        return False
    
    print(f"📄 使用配置文件: {data_yaml}")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ 错误: 未安装 ultralytics 库")
        print("   请运行: pip install ultralytics")
        return False
    
    # 训练参数 - 专门针对减少误识别优化
    training_args = {
        'data': data_yaml,
        'epochs': 50,           # 足够轮次
        'imgsz': 416,           # 中等分辨率
        'batch': 4,
        'workers': 0,
        'device': 'cpu',
        'name': 'apple_no_fp',  # "no false positives"
        'patience': 20,         # 更多耐心
        'save': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.0003,          # 低学习率
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,
        'box': 7.5,
        'cls': 1.0,             # 提高分类损失权重，增强分类能力
        'dfl': 1.5,
        
        # 数据增强 - 适度，避免生成混淆样本
        'hsv_h': 0.01,
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'fliplr': 0.3,          # 较低水平的翻转
        'degrees': 5.0,
        'translate': 0.05,
        'scale': 0.1,
        'shear': 0.5,
        'mosaic': 0.3,          # 降低马赛克概率，避免背景混乱
        'mixup': 0.0,           # 禁用mixup，防止类别混淆
        'copy_paste': 0.0,      # 禁用copy-paste
        
        # 验证参数
        'val': True,
        'plots': True,
    }
    
    print("\n⚙️  训练配置 (专门针对减少误识别):")
    print(f"  Epochs: {training_args['epochs']}")
    print(f"  图像尺寸: {training_args['imgsz']}")
    print(f"  分类损失权重: {training_args['cls']} (提高以增强苹果/非苹果区分能力)")
    print(f"  数据增强: 适度，避免产生混淆样本")
    print(f"  模型名称: {training_args['name']}")
    
    print("\n🚀 开始训练...")
    print("⏱️  预计时间: 40-80分钟 (取决于CPU性能)")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # 加载预训练模型
        model = YOLO('yolov8n.pt')
        
        # 开始训练
        results = model.train(**training_args)
        
        training_time = time.time() - start_time
        print(f"\n✅ 训练完成! 总耗时: {training_time/60:.1f} 分钟")
        
        # 复制最佳模型
        best_model_path = f"runs/detect/{training_args['name']}/weights/best.pt"
        if os.path.exists(best_model_path):
            output_model = "apple_no_fp.pt"
            shutil.copy2(best_model_path, output_model)
            print(f"\n💾 保存模型: {output_model}")
            
            # 验证模型
            print("\n🔍 验证模型性能...")
            val_results = model.val(data=data_yaml, device='cpu')
            
            if hasattr(val_results, 'box'):
                print(f"📊 验证结果:")
                print(f"  mAP50: {val_results.box.map50:.4f}")
                print(f"  mAP50-95: {val_results.box.map:.4f}")
                print(f"  精确率: {val_results.box.precision.mean():.4f}")
                print(f"  召回率: {val_results.box.recall.mean():.4f}")
            
            return output_model
        else:
            print(f"⚠️  未找到最佳模型文件 {best_model_path}")
            return None
            
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_new_model(model_path):
    """测试新模型在新增照片上的表现"""
    print_step(6, "测试新模型")
    
    if not model_path or not os.path.exists(model_path):
        print("❌ 错误: 模型文件不存在")
        return False
    
    new_photos_dir = r"C:\Users\李晨鑫\Desktop\apple（photo）"
    
    try:
        from ultralytics import YOLO
        import cv2
        
        # 加载模型
        print(f"🤖 加载模型: {os.path.basename(model_path)}")
        model = YOLO(model_path)
        
        # 获取测试图片
        image_exts = ['*.jpg', '*.jpeg', '*.png']
        image_paths = []
        for ext in image_exts:
            image_paths.extend(glob.glob(os.path.join(new_photos_dir, ext)))
        
        # 排除脚本文件
        image_paths = [p for p in image_paths if "detect_apple_to_video" not in p]
        
        print(f"📷 测试图片: {len(image_paths)} 张")
        
        if not image_paths:
            print("❌ 没有测试图片")
            return False
        
        # 保存结果的文件
        results_file = "misdetection_test_results.txt"
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"苹果检测模型误识别测试结果\n")
            f.write(f"模型: {model_path}\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
        
        # 在不同置信度下测试
        conf_thresholds = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
        
        print("\n🎯 测试不同置信度下的检测效果:")
        print("   目标: 每张图片检测到1个苹果 (因为每张只有1个苹果)")
        print("   问题: 检测到多个苹果 = 可能误识别\n")
        
        all_results = {}
        
        for conf in conf_thresholds:
            print(f"\n📊 置信度阈值: {conf}")
            
            misdetected_images = []
            correct_images = []
            no_detection_images = []
            
            for img_path in image_paths:
                img_name = os.path.basename(img_path)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                results = model(img, conf=conf, verbose=False)
                
                detections = 0
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        detections = len(result.boxes)
                
                # 评估
                if detections == 1:
                    correct_images.append(img_name)
                    status = "✅ 正确"
                elif detections == 0:
                    no_detection_images.append(img_name)
                    status = "⚠️  漏检"
                else:  # detections > 1
                    misdetected_images.append((img_name, detections))
                    status = f"❌ 误识别 ({detections}个)"
                
                print(f"   {img_name}: {status}")
            
            # 统计
            total = len(image_paths)
            misdetected_count = len(misdetected_images)
            correct_count = len(correct_images)
            no_detection_count = len(no_detection_images)
            
            print(f"   正确: {correct_count}/{total}, 误识别: {misdetected_count}/{total}, 漏检: {no_detection_count}/{total}")
            
            # 保存结果
            all_results[conf] = {
                'misdetected': misdetected_count,
                'correct': correct_count,
                'no_detection': no_detection_count,
                'misdetected_list': misdetected_images
            }
        
        # 分析最佳置信度
        print(f"\n{'='*60}")
        print("📈 置信度阈值分析:")
        
        best_conf = None
        best_score = -1
        
        for conf, stats in all_results.items():
            # 评分公式: 正确检测数 - 误识别数*2 (误识别更严重)
            score = stats['correct'] * 1 - stats['misdetected'] * 2
            print(f"  置信度 {conf}: 正确={stats['correct']}, 误识别={stats['misdetected']}, 漏检={stats['no_detection']}, 评分={score}")
            
            if score > best_score:
                best_score = score
                best_conf = conf
        
        print(f"\n🎯 推荐置信度阈值: {best_conf}")
        
        # 保存详细结果
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(f"\n推荐置信度阈值: {best_conf}\n\n")
            f.write(f"详细测试结果:\n")
            
            for conf, stats in all_results.items():
                f.write(f"\n置信度 {conf}:\n")
                f.write(f"  正确检测: {stats['correct']}\n")
                f.write(f"  误识别: {stats['misdetected']}\n")
                f.write(f"  漏检: {stats['no_detection']}\n")
                
                if stats['misdetected_list']:
                    f.write(f"  误识别图片:\n")
                    for img_name, detections in stats['misdetected_list']:
                        f.write(f"    - {img_name}: {detections}个检测框\n")
            
            f.write(f"\n{'='*60}\n")
            f.write(f"总结:\n")
            f.write(f"- 建议使用置信度阈值: {best_conf}\n")
            f.write(f"- 如果仍有误识别，考虑:\n")
            f.write(f"  1. 添加更多负样本进行训练\n")
            f.write(f"  2. 进一步降低置信度阈值\n")
            f.write(f"  3. 重新标注有问题的图片\n")
        
        print(f"\n📄 详细结果已保存到: {results_file}")
        
        # 显示最终评估
        best_stats = all_results[best_conf]
        print(f"\n{'='*60}")
        print("🏆 最终评估:")
        print(f"   使用置信度 {best_conf}:")
        print(f"   ✅ 正确检测: {best_stats['correct']}/{len(image_paths)}")
        print(f"   ❌ 误识别: {best_stats['misdetected']}/{len(image_paths)}")
        print(f"   ⚠️  漏检: {best_stats['no_detection']}/{len(image_paths)}")
        
        if best_stats['misdetected'] == 0:
            print(f"\n🎉 完美! 没有误识别!")
            return True
        elif best_stats['misdetected'] <= 2:
            print(f"\n👍 良好! 误识别很少")
            return True
        else:
            print(f"\n⚠️  仍有较多误识别，建议进一步优化")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print(f"{'='*80}")
    print("🍎 苹果检测模型误识别问题解决方案")
    print("目标: 解决将人、眼镜、头发误识别为苹果的问题")
    print(f"{'='*80}")
    
    # 切换到工作目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"工作目录: {script_dir}")
    
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
    
    # 步骤4: 添加负样本
    if not add_negative_samples():
        print("⚠️  添加负样本失败或警告")
    
    # 步骤5: 训练改进模型
    trained_model = train_misdetection_focused_model()
    if not trained_model:
        print("❌ 训练模型失败")
        return
    
    # 步骤6: 测试新模型
    test_passed = test_new_model(trained_model)
    
    print(f"\n{'='*80}")
    print("🧪 测试结果总结:")
    
    if test_passed:
        print("✅ 改进成功! 模型误识别问题已显著改善")
        print(f"📁 使用模型: {trained_model}")
        print(f"📄 详细结果: misdetection_test_results.txt")
        print(f"\n下一步:")
        print(f"1. 在Web应用中使用 {trained_model} 模型")
        print(f"2. 根据测试结果选择最佳置信度阈值")
        print(f"3. 继续监控实际使用中的误识别情况")
    else:
        print("⚠️  仍存在误识别问题，建议:")
        print(f"1. 检查 misdetection_test_results.txt 查看详细结果")
        print(f"2. 运行 python collect_negative_samples.py 收集更多负样本")
        print(f"3. 手动检查并修正数据集中有问题标签")
        print(f"4. 重新运行此脚本进行迭代优化")
    
    print(f"\n{'='*80}")
    print("🔧 其他可用命令:")
    print(f"   测试误识别: python test_misdetection.py")
    print(f"   收集负样本: python collect_negative_samples.py")
    print(f"   检查负样本比例: python check_negative_ratio.py")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()