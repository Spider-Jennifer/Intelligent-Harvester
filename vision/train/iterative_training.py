#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
迭代训练脚本 - 反复训练多次以提高准确率
每次训练都使用前一次的最佳模型作为基础
"""

import os
import time
import shutil
from pathlib import Path
from ultralytics import YOLO

def iterative_training(num_iterations=3, epochs_per_iteration=100):
    """
    执行多次迭代训练
    
    Args:
        num_iterations: 迭代次数
        epochs_per_iteration: 每次迭代的训练轮数
    """
    print("=" * 70)
    print(f"Iterative Training - {num_iterations} iterations")
    print(f"Epochs per iteration: {epochs_per_iteration}")
    print("=" * 70)
    
    # 检查数据集
    data_yaml = "apple_dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"Error: Config file not found: {data_yaml}")
        return
    
    # 选择初始基础模型
    base_models = ["yolov8n.pt", "apple_best.pt", "apple_sensitive.pt", "apple_improved.pt"]
    current_model = None
    
    for model in base_models:
        if os.path.exists(model):
            current_model = model
            print(f"Starting with base model: {current_model}")
            break
    
    if not current_model:
        print("Using default yolov8n.pt")
        current_model = "yolov8n.pt"
    
    # 迭代训练
    for iteration in range(1, num_iterations + 1):
        print(f"\n{'=' * 70}")
        print(f"Iteration {iteration}/{num_iterations}")
        print(f"{'=' * 70}")
        
        # 训练配置
        training_config = {
            'data': data_yaml,
            'epochs': epochs_per_iteration,
            'patience': 30,
            'batch': 16,
            'imgsz': 640,
            'device': '0' if os.path.exists('check_gpu.py') else 'cpu',
            'project': f'runs/iterative_training',
            'name': f'iteration_{iteration}',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.0001,
            'cos_lr': True,
            'save_period': 20,
            'plots': True,
            'verbose': True
        }
        
        print(f"Training configuration:")
        print(f"  Model: {current_model}")
        print(f"  Epochs: {training_config['epochs']}")
        print(f"  Learning rate: {training_config['lr0']}")
        
        start_time = time.time()
        
        try:
            # 加载模型
            model = YOLO(current_model)
            
            # 开始训练
            print(f"\nStarting training iteration {iteration}...")
            results = model.train(**training_config)
            
            # 更新当前模型为本次训练的最佳模型
            best_model_path = f"runs/iterative_training/iteration_{iteration}/weights/best.pt"
            if os.path.exists(best_model_path):
                # 保存为迭代模型
                iteration_model = f"apple_iteration_{iteration}.pt"
                shutil.copy(best_model_path, iteration_model)
                current_model = iteration_model
                
                print(f"\nIteration {iteration} completed!")
                print(f"Best model saved as: {iteration_model}")
                
                # 评估
                val_results = model.val()
                print(f"  mAP50-95: {val_results.box.map:.4f}")
                print(f"  mAP50: {val_results.box.map50:.4f}")
                print(f"  Precision: {val_results.box.p:.4f}")
                print(f"  Recall: {val_results.box.r:.4f}")
            
            end_time = time.time()
            iteration_time = (end_time - start_time) / 3600
            print(f"  Time: {iteration_time:.2f} hours")
            
        except Exception as e:
            print(f"Error in iteration {iteration}: {e}")
            continue
    
    # 最终总结
    print(f"\n{'=' * 70}")
    print("Iterative Training Completed!")
    print(f"{'=' * 70}")
    
    # 找到最佳模型
    best_map = 0
    best_model = None
    
    for i in range(1, num_iterations + 1):
        model_file = f"apple_iteration_{i}.pt"
        if os.path.exists(model_file):
            try:
                model = YOLO(model_file)
                results = model.val()
                map_score = results.box.map
                
                print(f"Iteration {i}: mAP50-95 = {map_score:.4f}")
                
                if map_score > best_map:
                    best_map = map_score
                    best_model = model_file
            except:
                print(f"Iteration {i}: Could not evaluate")
    
    if best_model:
        # 复制最佳模型
        shutil.copy(best_model, "apple_final_best.pt")
        print(f"\nBest model: {best_model} (mAP50-95: {best_map:.4f})")
        print(f"Final model saved as: apple_final_best.pt")
    
    print(f"\n{'=' * 70}")
    print("Next Steps:")
    print("1. Use apple_final_best.pt in your application")
    print("2. Test with confidence thresholds 0.15-0.35")
    print("3. Check runs/iterative_training/ for detailed results")
    print(f"{'=' * 70}")

def main():
    print("Iterative Training Menu")
    print("=" * 40)
    print("1. 3 iterations x 100 epochs (recommended)")
    print("2. 5 iterations x 80 epochs")
    print("3. 2 iterations x 150 epochs")
    print("4. Custom configuration")
    print("5. Exit")
    
    try:
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            iterative_training(num_iterations=3, epochs_per_iteration=100)
        elif choice == "2":
            iterative_training(num_iterations=5, epochs_per_iteration=80)
        elif choice == "3":
            iterative_training(num_iterations=2, epochs_per_iteration=150)
        elif choice == "4":
            try:
                iterations = int(input("Number of iterations: "))
                epochs = int(input("Epochs per iteration: "))
                iterative_training(num_iterations=iterations, epochs_per_iteration=epochs)
            except:
                print("Invalid input")
        elif choice == "5":
            print("Exiting...")
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()