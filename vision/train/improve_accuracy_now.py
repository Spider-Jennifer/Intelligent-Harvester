#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
立即提高模型准确率 - 优化版训练脚本
专注于提高检测准确率，使用最佳实践参数
"""

import os
import time
from ultralytics import YOLO

def main():
    print("=" * 60)
    print("立即提高苹果检测模型准确率")
    print("=" * 60)
    
    # 检查数据集
    data_yaml = "apple_dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"错误: 找不到配置文件 {data_yaml}")
        print("请确保数据集已准备")
        return
    
    print(f"使用数据集配置: {data_yaml}")
    
    # 检查数据集图片数量
    train_dir = "apple_dataset/images/train"
    val_dir = "apple_dataset/images/val"
    
    import glob
    train_images = glob.glob(os.path.join(train_dir, "*.jpg")) + glob.glob(os.path.join(train_dir, "*.png"))
    val_images = glob.glob(os.path.join(val_dir, "*.jpg")) + glob.glob(os.path.join(val_dir, "*.png"))
    
    print(f"训练集图片: {len(train_images)} 张")
    print(f"验证集图片: {len(val_images)} 张")
    
    if len(train_images) < 20:
        print("警告: 训练集图片数量较少，可能影响训练效果")
    
    # 选择最佳基础模型
    base_models = ["yolov8n.pt", "apple_best.pt", "apple_sensitive.pt"]
    available_models = []
    
    for model in base_models:
        if os.path.exists(model):
            available_models.append(model)
            print(f"找到可用基础模型: {model}")
    
    if not available_models:
        print("错误: 没有找到可用的基础模型")
        return
    
    # 使用最佳的基础模型
    base_model = available_models[0]
    print(f"\n使用基础模型: {base_model}")
    
    # 训练参数 - 针对提高准确率优化
    training_args = {
        'data': data_yaml,
        'epochs': 100,           # 足够的训练轮数
        'imgsz': 640,           # 标准分辨率
        'batch': 4,             # 小批次，适合CPU
        'workers': 0,           # Windows上设为0
        'device': 'cpu',        # 使用CPU
        'name': 'apple_accuracy_improved',
        'patience': 25,         # 早停耐心值
        'save': True,
        'pretrained': True,
        'optimizer': 'AdamW',   # AdamW优化器，更好的收敛性
        'lr0': 0.0003,          # 较低的学习率，更稳定
        'lrf': 0.01,           # 学习率衰减
        'momentum': 0.937,
        'weight_decay': 0.0005, # 权重衰减，防止过拟合
        'warmup_epochs': 5,     # 热身epoch
        'box': 7.5,            # 边界框损失权重
        'cls': 0.5,            # 分类损失权重
        'dfl': 1.5,            # DFL损失权重
        
        # 数据增强 - 提高模型泛化能力
        'hsv_h': 0.015,        # 色调增强
        'hsv_s': 0.7,          # 饱和度增强
        'hsv_v': 0.4,          # 亮度增强
        'fliplr': 0.5,         # 水平翻转
        'mosaic': 0.8,         # 马赛克增强（高值提高泛化）
        'degrees': 15.0,       # 旋转增强
        'translate': 0.1,      # 平移增强
        'scale': 0.2,          # 缩放增强
        'shear': 0.0,          # 剪切增强
        'perspective': 0.0001, # 透视变换
        'copy_paste': 0.0,     # 复制粘贴增强
        'erasing': 0.4,        # 随机擦除
        'crop_fraction': 0.8,  # 随机裁剪
        
        # 其他优化参数
        'close_mosaic': 10,    # 最后10个epoch关闭马赛克
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,           # 启用验证
        'plots': True,         # 生成训练图表
    }
    
    print("\n开始训练...")
    print("训练参数:")
    for key, value in training_args.items():
        if key not in ['data', 'pretrained']:
            print(f"  {key}: {value}")
    
    print("\n预计训练时间: 30-60分钟 (取决于CPU性能)")
    print("按 Ctrl+C 可以中断训练")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # 加载模型
        model = YOLO(base_model)
        
        # 开始训练
        results = model.train(**training_args)
        
        training_time = time.time() - start_time
        print(f"\n训练完成! 总耗时: {training_time/60:.1f} 分钟")
        
        # 获取最佳模型路径
        best_model_path = "runs/detect/apple_accuracy_improved/weights/best.pt"
        if os.path.exists(best_model_path):
            # 复制到项目根目录
            import shutil
            improved_model = "apple_accuracy_improved.pt"
            shutil.copy(best_model_path, improved_model)
            print(f"\n最佳模型已保存为: {improved_model}")
            print(f"原始位置: {best_model_path}")
            
            # 验证模型性能
            print("\n验证模型性能...")
            val_results = model.val(
                data=data_yaml,
                imgsz=640,
                batch=4,
                conf=0.25,
                iou=0.45,
                device='cpu'
            )
            
            if hasattr(val_results, 'box'):
                print(f"\n验证结果:")
                print(f"  mAP50: {val_results.box.map50:.4f}")
                print(f"  mAP50-95: {val_results.box.map:.4f}")
                print(f"  精确率: {val_results.box.precision.mean():.4f}")
                print(f"  召回率: {val_results.box.recall.mean():.4f}")
                
                # 给出使用建议
                print("\n使用建议:")
                print("1. 在Web应用中使用 'apple_accuracy_improved.pt' 模型")
                print("2. 建议置信度阈值: 0.15-0.25")
                print("3. 如果检测框过多，适当提高置信度阈值")
                print("4. 如果漏检较多，适当降低置信度阈值")
            else:
                print("无法获取验证结果")
        else:
            print("警告: 未找到最佳模型文件")
            
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n训练过程结束")

if __name__ == "__main__":
    main()