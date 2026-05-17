#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试训练好的苹果检测模型
"""

import os
import cv2
import glob
from pathlib import Path
from ultralytics import YOLO

def test_model_on_images():
    """在图片上测试模型"""
    print("=" * 60)
    print("测试训练好的苹果检测模型")
    print("=" * 60)
    
    # 查找训练好的模型
    model_candidates = [
        "runs/detect/apple_detection_simple/weights/best.pt",
        "apple_best.pt",
        "yolov8n.pt"
    ]
    
    model_path = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            model_path = candidate
            print(f"找到模型: {model_path}")
            break
    
    if not model_path:
        print("错误: 找不到任何模型文件")
        print("请先运行训练脚本: python train_apple_simple.py")
        return
    
    try:
        # 加载模型
        model = YOLO(model_path)
        print(f"模型加载成功: {model_path}")
        print()
        
        # 测试图片目录
        test_dirs = [
            r"c:\Users\李晨鑫\Desktop\yolov8\YOLO-v8-app\dataset\images\val",
            r"c:\Users\李晨鑫\Desktop\yolov8\YOLO-v8-app\dataset\images\train"
        ]
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                print(f"测试目录: {test_dir}")
                
                # 获取测试图片
                test_files = glob.glob(os.path.join(test_dir, "*.jpg"))
                test_files += glob.glob(os.path.join(test_dir, "*.png"))
                
                if test_files:
                    print(f"找到 {len(test_files)} 张测试图片")
                    print("测试前5张图片...")
                    print()
                    
                    for i, img_path in enumerate(test_files[:5]):
                        img_name = os.path.basename(img_path)
                        print(f"测试图片 {i+1}: {img_name}")
                        
                        # 读取图片
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"  无法读取图片: {img_path}")
                            continue
                        
                        # 进行预测
                        results = model(img, conf=0.5, verbose=False)
                        
                        # 显示结果
                        if results and len(results) > 0:
                            result = results[0]
                            if result.boxes is not None:
                                boxes = result.boxes.xyxy.cpu().numpy()
                                confs = result.boxes.conf.cpu().numpy()
                                
                                print(f"  检测到 {len(boxes)} 个苹果")
                                
                                # 绘制检测结果
                                for box, conf in zip(boxes, confs):
                                    x1, y1, x2, y2 = map(int, box)
                                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    label = f"Apple: {conf:.2f}"
                                    cv2.putText(img, label, (x1, y1-10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            else:
                                print("  未检测到苹果")
                        else:
                            print("  未检测到苹果")
                        
                        # 保存结果
                        output_dir = "test_results"
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f"result_{img_name}")
                        cv2.imwrite(output_path, img)
                        print(f"  结果保存到: {output_path}")
                        print()
                    
                    print(f"测试完成，结果保存在: {output_dir}")
                    print()
                else:
                    print("该目录没有图片")
                    print()
            else:
                print(f"测试目录不存在: {test_dir}")
                print()
        
        print("=" * 60)
        print("测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def test_model_on_camera():
    """在摄像头上测试模型"""
    print("=" * 60)
    print("在摄像头上测试苹果检测模型")
    print("=" * 60)
    
    # 查找训练好的模型
    model_candidates = [
        "runs/detect/apple_detection_simple/weights/best.pt",
        "apple_best.pt"
    ]
    
    model_path = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            model_path = candidate
            print(f"找到模型: {model_path}")
            break
    
    if not model_path:
        print("错误: 找不到训练好的模型")
        print("请先运行训练脚本")
        return
    
    try:
        # 加载模型
        model = YOLO(model_path)
        
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        print("摄像头已打开，按 'q' 退出")
        print("按 's' 保存当前帧")
        
        frame_count = 0
        save_count = 0
        
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 每2帧处理一次（提高性能）
            if frame_count % 2 == 0:
                # 进行预测
                results = model(frame, conf=0.5, verbose=False)
                
                # 绘制检测结果
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        
                        for box, conf in zip(boxes, confs):
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"Apple: {conf:.2f}"
                            cv2.putText(frame, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # 显示检测数量
                        cv2.putText(frame, f"Detections: {len(boxes)}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 显示帧
            cv2.imshow('Apple Detection Test', frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                save_count += 1
                save_path = f"camera_test_{save_count}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"保存帧到: {save_path}")
        
        # 清理
        cap.release()
        cv2.destroyAllWindows()
        
        print("摄像头测试完成")
        
    except Exception as e:
        print(f"摄像头测试过程中出现错误: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "camera":
        test_model_on_camera()
    else:
        test_model_on_images()