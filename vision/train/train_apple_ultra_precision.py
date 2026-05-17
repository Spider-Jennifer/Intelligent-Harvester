#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
超精度苹果检测模型训练
专门解决眼镜、人脸等物体的误识别问题，强化苹果特征学习
"""

import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import yaml

def check_dataset_and_negative_samples():
    """检查数据集和负样本情况"""
    print("=" * 80)
    print("数据集与负样本检查")
    print("=" * 80)
    
    # 检查数据集目录
    data_yaml = "apple_dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"错误: 找不到数据集配置文件 {data_yaml}")
        return False
    
    # 检查训练标签目录
    train_labels_dir = "apple_dataset/labels/train"
    if not os.path.exists(train_labels_dir):
        print(f"错误: 找不到训练标签目录 {train_labels_dir}")
        return False
    
    # 统计标签文件
    label_files = [f for f in os.listdir(train_labels_dir) if f.endswith('.txt')]
    empty_labels = 0
    non_empty_labels = 0
    
    for label_file in label_files:
        file_path = os.path.join(train_labels_dir, label_file)
        if os.path.getsize(file_path) == 0:
            empty_labels += 1
        else:
            non_empty_labels += 1
    
    total_labels = len(label_files)
    negative_ratio = empty_labels / total_labels * 100 if total_labels > 0 else 0
    
    print(f"\n训练集标签统计:")
    print(f"  总标签文件: {total_labels}")
    print(f"  非空标签（有苹果）: {non_empty_labels}")
    print(f"  空标签（负样本）: {empty_labels}")
    print(f"  负样本比例: {negative_ratio:.1f}%")
    
    # 负样本比例建议
    if negative_ratio < 10:
        print(f"  ⚠️  警告: 负样本比例较低，建议至少20%以减少误识别")
        print(f"  建议运行 collect_negative_samples.py 收集更多负样本")
    elif negative_ratio < 20:
        print(f"  ⚠️  注意: 负样本比例适中，可以进一步提高")
    else:
        print(f"  ✅ 良好: 负样本比例合适")
    
    # 检查验证集
    val_labels_dir = "apple_dataset/labels/val"
    if os.path.exists(val_labels_dir):
        val_files = [f for f in os.listdir(val_labels_dir) if f.endswith('.txt')]
        val_empty = sum(1 for f in val_files if os.path.getsize(os.path.join(val_labels_dir, f)) == 0)
        print(f"\n验证集标签统计:")
        print(f"  总标签文件: {len(val_files)}")
        print(f"  空标签（负样本）: {val_empty}")
    
    return True

def prepare_advanced_training_config():
    """准备高级训练配置"""
    print("\n" + "=" * 80)
    print("超精度苹果检测模型训练配置")
    print("=" * 80)
    
    # 高级训练配置 - 专门针对误识别问题优化
    training_config = {
        'data': 'apple_dataset/data.yaml',
        'epochs': 50,                     # 更多epoch获得稳定学习
        'imgsz': 640,                     # 高分辨率，更好特征学习
        'batch': 4,                       # 小批量，更稳定
        'workers': 0,                      # Windows上设为0避免问题
        'device': 'cpu',                   # 使用CPU训练
        'name': f'apple_ultra_precision_{datetime.now().strftime("%Y%m%d_%H%M")}',
        'patience': 20,                    # 早停耐心
        'save': True,
        'save_period': 10,
        'pretrained': True,                # 使用预训练权重
        'optimizer': 'AdamW',              # 更好的优化器
        'lr0': 0.0002,                     # 更低学习率，稳定训练
        'lrf': 0.0001,                     # 最终学习率因子
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,                # 更长热身
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # 损失函数权重 - 关键调整
        'box': 5.0,                        # 适度降低框损失权重
        'cls': 2.0,                        # 大幅提高分类损失权重，减少误识别
        'dfl': 1.5,
        
        # 数据增强 - 针对误识别优化
        'hsv_h': 0.015,                    # 适度色调增强，提高颜色鲁棒性
        'hsv_s': 0.8,                      # 较大饱和度增强，强调苹果颜色特征
        'hsv_v': 0.4,                      # 适度明度增强
        
        # 空间变换 - 谨慎使用
        'degrees': 2.0,                    # 小角度旋转
        'translate': 0.05,                 # 小幅度平移
        'scale': 0.2,                      # 适度缩放
        'shear': 0.5,                      # 小幅度剪切
        
        # 特殊增强
        'perspective': 0.0005,             # 极小透视变换
        'flipud': 0.0,                     # 禁用上下翻转（苹果通常不会倒置）
        'fliplr': 0.3,                     # 适度水平翻转
        
        # 复杂增强 - 谨慎使用
        'mosaic': 0.2,                     # 适度mosaic增强
        'mixup': 0.0,                      # 禁用mixup，避免类别混淆
        'copy_paste': 0.0,                 # 禁用复制粘贴
        
        # 其他增强
        'erasing': 0.3,                    # 随机擦除，提高鲁棒性
        'augment': True,                   # 启用增强
        'rect': False,                     # 禁用矩形训练
        'cos_lr': True,                    # 使用余弦退火学习率
        'close_mosaic': 15,                # 最后15个epoch关闭mosaic
        'fraction': 1.0,                   # 使用全部数据
        
        # 验证设置
        'val': True,
        'split': 'val',
        'conf': 0.4,                       # 验证时使用更高置信度阈值
        'iou': 0.6,                        # 更高IoU阈值，要求更精确匹配
        'max_det': 10,                     # 限制最大检测数，防止过度检测
        
        # 输出设置
        'plots': True,
        'save_json': False,
        'save_hybrid': False,
        'verbose': True,
        'exist_ok': True,
    }
    
    print("\n训练配置详情:")
    print(f"  训练轮数: {training_config['epochs']}")
    print(f"  图像尺寸: {training_config['imgsz']}")
    print(f"  批次大小: {training_config['batch']}")
    print(f"  学习率: {training_config['lr0']} (非常低，稳定训练)")
    print(f"  分类损失权重: {training_config['cls']} (非常高，显著减少误识别)")
    print(f"  验证置信度阈值: {training_config['conf']} (较高，严格评估)")
    print(f"  验证IoU阈值: {training_config['iou']} (较高，要求精确匹配)")
    print(f"  最大检测数: {training_config['max_det']} (限制过度检测)")
    
    print("\n数据增强策略:")
    print(f"  饱和度增强: {training_config['hsv_s']} (强化颜色特征)")
    print(f"  随机擦除: {training_config['erasing']} (提高鲁棒性)")
    print(f"  余弦退火学习率: {'启用' if training_config['cos_lr'] else '禁用'}")
    print(f"  混合增强: 禁用 (避免类别混淆)")
    
    return training_config

def verify_model_compatibility():
    """验证模型兼容性"""
    print("\n" + "=" * 80)
    print("模型兼容性验证")
    print("=" * 80)
    
    try:
        # 尝试加载预训练模型
        print("检查预训练模型 yolov8n.pt...")
        model = YOLO('yolov8n.pt')
        
        # 检查模型信息
        print(f"  模型类型: {model.__class__.__name__}")
        print(f"  类别数量: {len(model.names)}")
        print(f"  模型结构: YOLOv8n")
        
        # 测试轻量级推理
        print("  进行快速兼容性测试...")
        test_img = None
        
        # 尝试找一张测试图片
        test_dirs = [
            "apple_dataset/images/train",
            "apple_dataset/images/val"
        ]
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                for ext in ['.jpg', '.jpeg', '.png']:
                    test_files = list(Path(test_dir).glob(f"*{ext}"))
                    if test_files:
                        test_img = str(test_files[0])
                        break
                if test_img:
                    break
        
        if test_img:
            print(f"  使用测试图片: {os.path.basename(test_img)}")
            results = model(test_img, conf=0.3, verbose=False)
            print(f"  兼容性测试通过")
        else:
            print(f"  未找到测试图片，跳过快速测试")
        
        return True
    
    except Exception as e:
        print(f"  模型兼容性验证失败: {e}")
        print(f"  可能原因: 模型文件损坏或格式不兼容")
        return False

def train_ultra_precision_model():
    """训练超精度苹果检测模型"""
    print("\n" + "=" * 80)
    print("开始训练超精度苹果检测模型")
    print("=" * 80)
    
    start_time = time.time()
    
    # 1. 检查数据集
    if not check_dataset_and_negative_samples():
        print("数据集检查失败，请确保数据集完整")
        return False
    
    # 2. 验证模型兼容性
    if not verify_model_compatibility():
        print("模型兼容性验证失败")
        return False
    
    # 3. 准备训练配置
    training_config = prepare_advanced_training_config()
    
    # 4. 加载模型
    print("\n" + "=" * 80)
    print("加载预训练模型")
    print("=" * 80)
    
    try:
        model = YOLO('yolov8n.pt')
        print("预训练模型加载成功")
    except Exception as e:
        print(f"加载预训练模型失败: {e}")
        
        # 尝试下载模型
        print("尝试下载预训练模型...")
        try:
            model = YOLO('yolov8n.pt')
            print("预训练模型下载成功")
        except Exception as e2:
            print(f"下载预训练模型失败: {e2}")
            return False
    
    # 5. 开始训练
    print("\n" + "=" * 80)
    print("开始超精度训练")
    print("=" * 80)
    print("注意: 训练可能需要60-90分钟，请耐心等待")
    print("关键改进: 分类损失权重大幅提高，显著减少误识别")
    print("-" * 80)
    
    try:
        results = model.train(**training_config)
        
        training_time = time.time() - start_time
        print(f"\n训练完成! 总耗时: {training_time/60:.1f} 分钟")
        
        # 保存最终模型
        save_final_model(training_config['name'])
        
        return True
    
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        print("建议:")
        print("1. 检查数据集路径是否正确")
        print("2. 确保有足够的内存空间")
        print("3. 尝试减少批次大小 (batch=2)")
        print("4. 尝试减少图像尺寸 (imgsz=320)")
        return False

def save_final_model(experiment_name):
    """保存最终模型并生成报告"""
    print("\n" + "=" * 80)
    print("保存最终模型与生成报告")
    print("=" * 80)
    
    # 查找最佳模型
    best_model_path = f"runs/detect/{experiment_name}/weights/best.pt"
    last_model_path = f"runs/detect/{experiment_name}/weights/last.pt"
    
    final_model_name = "apple_ultra_precision.pt"
    
    # 复制最佳模型
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, final_model_name)
        print(f"✅ 最佳模型已保存: {final_model_name}")
        print(f"   原始路径: {best_model_path}")
        
        # 获取文件大小
        file_size_mb = os.path.getsize(final_model_name) / (1024 * 1024)
        print(f"   文件大小: {file_size_mb:.1f} MB")
        
        # 测试最终模型
        test_final_model(final_model_name)
        
        # 生成训练报告
        generate_training_report(experiment_name, final_model_name)
    else:
        print(f"⚠️  警告: 找不到最佳模型 {best_model_path}")
        
        # 尝试复制最终模型
        if os.path.exists(last_model_path):
            shutil.copy2(last_model_path, final_model_name)
            print(f"使用最终模型替代: {final_model_name}")
        else:
            print("无法找到任何训练好的模型")

def test_final_model(model_path):
    """测试最终模型性能"""
    print("\n快速测试最终模型...")
    
    try:
        model = YOLO(model_path)
        
        # 测试图片目录
        test_dirs = [
            "apple_dataset/images/val",
            "C:\\Users\\李晨鑫\\Desktop\\苹果照片检测素材"
        ]
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                print(f"\n测试目录: {test_dir}")
                
                # 获取前3张图片
                image_extensions = ['*.jpg', '*.jpeg', '*.png']
                test_images = []
                for ext in image_extensions:
                    test_images.extend(list(Path(test_dir).glob(ext)))
                
                test_images = test_images[:3]  # 只测试前3张
                
                if test_images:
                    for img_path in test_images:
                        print(f"  图片: {img_path.name}")
                        
                        # 在不同置信度下测试
                        for conf in [0.2, 0.3, 0.4, 0.5]:
                            results = model(str(img_path), conf=conf, verbose=False)
                            
                            if results and len(results) > 0:
                                result = results[0]
                                if result.boxes is not None:
                                    detections = len(result.boxes)
                                    print(f"    置信度 {conf}: {detections} 个苹果")
                                else:
                                    print(f"    置信度 {conf}: 0 个苹果")
                            else:
                                print(f"    置信度 {conf}: 0 个苹果")
                else:
                    print(f"  目录中没有图片")
    except Exception as e:
        print(f"测试模型时出错: {e}")

def generate_training_report(experiment_name, model_path):
    """生成训练报告"""
    print("\n生成训练报告...")
    
    report_content = f"""# 超精度苹果检测模型训练报告

## 训练信息
- 训练时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 实验名称: {experiment_name}
- 模型文件: {model_path}

## 训练配置亮点
1. **分类损失权重大幅提高** (cls=2.0) - 显著减少误识别
2. **学习率非常低** (lr0=0.0002) - 稳定训练，避免过拟合
3. **验证置信度阈值较高** (conf=0.4) - 严格评估，减少假阳性
4. **验证IoU阈值较高** (iou=0.6) - 要求精确匹配
5. **最大检测数限制** (max_det=10) - 防止过度检测

## 数据增强策略
- **颜色增强**: 强调苹果的颜色特征
- **随机擦除**: 提高模型鲁棒性
- **禁用混合增强**: 避免类别混淆
- **余弦退火学习率**: 让学习率平稳下降

## 负样本使用
- 空标签文件作为负样本
- 帮助模型学习区分苹果和非苹果物体
- 特别针对眼镜、人脸等常见误识别物体

## 预期效果
1. ✅ **大幅减少误识别** - 特别是眼镜、人脸等非苹果物体
2. ✅ **保持高检测率** - 对真实苹果的检测能力不受影响
3. ✅ **更高置信度要求** - 模型对识别结果更有信心

## 使用建议
1. **推理时使用较高置信度阈值** (0.3-0.4)
2. **如果仍有误识别，可进一步提高阈值**
3. **可在苹果照片检测素材上测试效果**

## 后续优化建议
1. 收集更多负样本（特别是包含眼镜的图片）
2. 考虑使用更大模型 (yolov8m) 获得更好特征提取
3. 手动修正部分标注，确保精度
"""
    
    report_path = "training_report_ultra_precision.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"训练报告已生成: {report_path}")

def main():
    """主函数"""
    print("=" * 80)
    print("超精度苹果检测模型训练")
    print("=" * 80)
    print("专门解决眼镜、人脸等物体的误识别问题")
    print("通过大幅提高分类损失权重和优化训练策略")
    print("显著提升苹果特征识别能力，减少误识别")
    print("=" * 80)
    
    # 检查是否有足够负样本
    print("\n第一步: 检查数据集")
    
    # 询问是否先收集负样本
    print("\n建议: 在开始训练前，确保有足够的负样本（至少20%）")
    print("是否先运行负样本收集工具? (y/n)")
    choice = input().strip().lower()
    
    if choice == 'y':
        print("请先运行 python collect_negative_samples.py")
        print("收集完负样本后重新运行本脚本")
        return
    
    # 开始训练
    print("\n开始超精度训练...")
    success = train_ultra_precision_model()
    
    if success:
        print("\n" + "=" * 80)
        print("训练成功完成!")
        print("=" * 80)
        print("下一步:")
        print(f"1. 测试新模型: python test_misdetection.py")
        print(f"2. 更新检测脚本: 修改 detect_apple_improved.py 中的模型路径")
        print(f"3. 生成新检测视频: 运行检测脚本查看效果")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("训练失败!")
        print("=" * 80)
        print("请检查错误信息并修正后重新尝试")

if __name__ == "__main__":
    main()