#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IoU阈值调整演示脚本
展示不同IoU阈值对检测结果的影响
"""

# import numpy as np  # 移除依赖，直接演示概念

def demonstrate_iou_thresholds():
    """
    演示不同IoU阈值对检测结果的影响
    """
    print("=" * 60)
    print("IoU阈值调整演示")
    print("=" * 60)
    
    # 模拟检测结果
    boxes = [
        {'box': [100, 100, 200, 200], 'confidence': 0.9, 'class': 'apple'},
        {'box': [110, 110, 210, 210], 'confidence': 0.8, 'class': 'apple'},
        {'box': [300, 300, 400, 400], 'confidence': 0.7, 'class': 'apple'},
        {'box': [310, 310, 410, 410], 'confidence': 0.6, 'class': 'apple'},
    ]
    
    def calculate_iou(box1, box2):
        """计算两个边界框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union
    
    def apply_nms(boxes, iou_threshold):
        """应用非极大值抑制"""
        if not boxes:
            return []
        
        # 按置信度排序
        boxes_sorted = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
        kept = []
        
        while boxes_sorted:
            current = boxes_sorted.pop(0)
            kept.append(current)
            
            # 移除与当前框IoU过高的其他框
            boxes_sorted = [box for box in boxes_sorted 
                           if calculate_iou(current['box'], box['box']) < iou_threshold]
        
        return kept
    
    # 测试不同IoU阈值
    iou_thresholds = [0.3, 0.5, 0.7]
    
    print("\n原始检测结果:")
    for i, box in enumerate(boxes):
        print(f"  框{i+1}: {box['box']}, 置信度: {box['confidence']:.2f}")
    
    print("\n" + "=" * 60)
    
    for threshold in iou_thresholds:
        kept_boxes = apply_nms(boxes, threshold)
        
        print(f"\nIoU阈值: {threshold}")
        print(f"保留的检测框数量: {len(kept_boxes)}")
        print("保留的检测框:")
        for i, box in enumerate(kept_boxes):
            print(f"  框{i+1}: {box['box']}, 置信度: {box['confidence']:.2f}")
        
        # 计算被移除的框
        removed_count = len(boxes) - len(kept_boxes)
        print(f"被移除的重复框: {removed_count}")
    
    print("\n" + "=" * 60)
    print("IoU阈值调整建议:")
    print("1. 密集目标检测: 降低IoU阈值 (0.3-0.4)")
    print("2. 稀疏目标检测: 提高IoU阈值 (0.6-0.7)")
    print("3. 通用场景: 使用默认值 (0.45-0.5)")
    print("4. 小目标检测: 适当降低阈值避免漏检")
    print("5. 大目标检测: 可以使用较高阈值")

def modify_yolo_iou_settings():
    """
    修改YOLOv8的IoU相关设置
    """
    print("\n" + "=" * 60)
    print("YOLOv8 IoU设置修改方法")
    print("=" * 60)
    
    print("\n1. 在推理时修改IoU阈值:")
    print("```python")
    print("from ultralytics import YOLO")
    print("")
    print("# 加载模型")
    print("model = YOLO('best.pt')")
    print("")
    print("# 推理时设置IoU阈值")
    print("results = model.predict(")
    print("    source='image.jpg',")
    print("    conf=0.25,        # 置信度阈值")
    print("    iou=0.45,         # IoU阈值 (NMS)")
    print("    max_det=1000      # 最大检测数量")
    print(")")
    print("```")
    
    print("\n2. 在训练时修改IoU阈值:")
    print("```python")
    print("# 训练配置")
    print("results = model.train(")
    print("    data='dataset.yaml',")
    print("    epochs=100,")
    print("    iou=0.45,         # NMS IoU阈值")
    print("    box_iou=0.7,      # 边界框IoU阈值")
    print("    plots=True")
    print(")")
    print("```")
    
    print("\n3. 针对苹果检测的推荐设置:")
    print("- 置信度阈值: 0.25-0.35 (苹果特征明显)")
    print("- IoU阈值: 0.4-0.5 (避免重叠苹果被误删)")
    print("- 最大检测数: 100 (一张图片可能有多个苹果)")

if __name__ == "__main__":
    demonstrate_iou_thresholds()
    modify_yolo_iou_settings()