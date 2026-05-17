#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成训练指标和延迟数据的GIF动画视频
展示精确率、召回率、mAP和延迟等数据的动态变化
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import imageio
from datetime import datetime

# ============= 配置参数 =============
# 训练结果CSV文件路径模式
RESULTS_PATTERN = "runs/detect/*/results.csv"
# 延迟测试图片数量（用于快速测试）
LATENCY_TEST_IMAGES = 10
# 输出GIF文件名
OUTPUT_GIF = "metrics_latency_animation.gif"
# GIF帧率
GIF_FPS = 5
# 是否重新运行延迟测试（True：运行测试，False：使用现有数据）
RERUN_LATENCY_TEST = True
# ===================================

def load_training_results():
    """加载所有训练结果CSV文件"""
    csv_files = glob.glob(RESULTS_PATTERN)
    print(f"找到 {len(csv_files)} 个训练结果文件")
    
    results = {}
    for csv_file in csv_files:
        model_name = os.path.basename(os.path.dirname(csv_file))
        try:
            df = pd.read_csv(csv_file)
            # 确保有epoch列
            if 'epoch' not in df.columns:
                df['epoch'] = range(1, len(df) + 1)
            results[model_name] = df
            print(f"  已加载 {model_name}: {len(df)} 个epoch")
        except Exception as e:
            print(f"  加载 {csv_file} 失败: {e}")
    
    return results

def run_latency_test():
    """运行快速延迟测试，获取各模型延迟数据"""
    from ultralytics import YOLO
    import time
    import random
    
    # 查找可用的模型文件
    model_patterns = [
        "apple_sensitive.pt",
        "apple_improved.pt", 
        "apple_best.pt",
        "apple_quick.pt",
        "yolov8n.pt"
    ]
    
    available_models = []
    for model_path in model_patterns:
        if os.path.exists(model_path):
            available_models.append(model_path)
        else:
            # 检查runs目录中的模型
            alt_path = f"runs/detect/{model_path.replace('.pt', '')}/weights/best.pt"
            if os.path.exists(alt_path):
                available_models.append(alt_path)
    
    print(f"找到 {len(available_models)} 个模型进行延迟测试")
    
    # 获取测试图片
    test_images = []
    apple_photos_dir = "apple_photos"
    if os.path.exists(apple_photos_dir):
        image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        for ext in image_exts:
            test_images.extend(glob.glob(os.path.join(apple_photos_dir, ext)))
    
    if not test_images:
        # 使用训练集中的图片作为备选
        train_dir = r"C:\Users\李晨鑫\Desktop\yolov8\apple_dataset\images\train"
        if os.path.exists(train_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                test_images.extend(glob.glob(os.path.join(train_dir, ext)))
    
    if len(test_images) > LATENCY_TEST_IMAGES:
        test_images = random.sample(test_images, LATENCY_TEST_IMAGES)
    
    print(f"使用 {len(test_images)} 张图片进行延迟测试")
    
    latency_results = {}
    for model_path in available_models:
        model_name = os.path.basename(model_path).replace('.pt', '')
        print(f"  测试模型: {model_name}")
        
        try:
            # 加载模型
            start_time = time.time()
            model = YOLO(model_path)
            load_time = time.time() - start_time
            
            # 预热
            warmup_images = min(3, len(test_images))
            for i in range(warmup_images):
                model(test_images[i], verbose=False)
            
            # 正式测试
            inference_times = []
            for img_path in test_images:
                start = time.time()
                results = model(img_path, verbose=False)
                inference_times.append(time.time() - start)
            
            # 计算统计信息
            avg_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            min_time = np.min(inference_times)
            max_time = np.max(inference_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            latency_results[model_name] = {
                'avg_inference_time': avg_time,
                'std_inference_time': std_time,
                'min_inference_time': min_time,
                'max_inference_time': max_time,
                'fps': fps,
                'load_time': load_time,
                'test_images_count': len(test_images)
            }
            
            print(f"    平均延迟: {avg_time:.4f}秒, FPS: {fps:.2f}")
            
        except Exception as e:
            print(f"    测试失败: {e}")
    
    return latency_results

def create_animation_frames(training_results, latency_results):
    """创建动画帧序列"""
    frames = []
    
    # 确定最大epoch数
    max_epochs = 0
    for df in training_results.values():
        if len(df) > max_epochs:
            max_epochs = len(df)
    
    if max_epochs == 0:
        print("没有训练数据可用")
        return frames
    
    # 获取模型名称列表
    model_names = list(training_results.keys())
    latency_model_names = list(latency_results.keys())
    
    # 设置颜色映射
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    
    # 创建图形和子图
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3)
    
    # 子图定义
    ax1 = fig.add_subplot(gs[0, 0])  # 精确率
    ax2 = fig.add_subplot(gs[0, 1])  # 召回率
    ax3 = fig.add_subplot(gs[0, 2])  # mAP50
    ax4 = fig.add_subplot(gs[1, :])  # 延迟对比
    ax5 = fig.add_subplot(gs[2, :])  # 数据表格
    
    # 设置全局标题
    fig.suptitle('YOLOv8 苹果检测模型性能指标动态展示', fontsize=16, fontweight='bold', y=0.98)
    
    # 初始化动画帧列表
    print(f"生成动画帧，共 {max_epochs} 帧...")
    
    for epoch in range(1, max_epochs + 1):
        # 清空所有子图
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()
        
        # 设置子图标题
        ax1.set_title(f'精确率 (Precision) - Epoch {epoch}/{max_epochs}', fontsize=12)
        ax2.set_title(f'召回率 (Recall) - Epoch {epoch}/{max_epochs}', fontsize=12)
        ax3.set_title(f'mAP50 - Epoch {epoch}/{max_epochs}', fontsize=12)
        ax4.set_title('模型推理延迟对比', fontsize=12)
        ax5.set_title('当前指标数值', fontsize=12)
        
        # 绘制训练指标曲线（截至当前epoch）
        for idx, (model_name, df) in enumerate(training_results.items()):
            color = colors[idx]
            df_subset = df[df['epoch'] <= epoch]
            
            if len(df_subset) > 0:
                # 精确率
                if 'metrics/precision(B)' in df.columns:
                    ax1.plot(df_subset['epoch'], df_subset['metrics/precision(B)'], 
                            color=color, linewidth=2, label=model_name, marker='o', markersize=3)
                
                # 召回率
                if 'metrics/recall(B)' in df.columns:
                    ax2.plot(df_subset['epoch'], df_subset['metrics/recall(B)'], 
                            color=color, linewidth=2, label=model_name, marker='s', markersize=3)
                
                # mAP50
                if 'metrics/mAP50(B)' in df.columns:
                    ax3.plot(df_subset['epoch'], df_subset['metrics/mAP50(B)'], 
                            color=color, linewidth=2, label=model_name, marker='^', markersize=3)
        
        # 设置坐标轴标签
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Precision')
        ax1.set_ylim(0.5, 1.05)
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Recall')
        ax2.set_ylim(0.5, 1.05)
        ax2.grid(True, alpha=0.3)
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('mAP50')
        ax3.set_ylim(0.5, 1.05)
        ax3.grid(True, alpha=0.3)
        
        # 添加图例（只显示一次）
        if epoch == max_epochs:
            ax1.legend(loc='lower right', fontsize=8)
            ax2.legend(loc='lower right', fontsize=8)
            ax3.legend(loc='lower right', fontsize=8)
        
        # 绘制延迟对比柱状图
        if latency_results:
            latency_models = list(latency_results.keys())
            avg_times = [latency_results[m]['avg_inference_time'] for m in latency_models]
            fps_values = [latency_results[m]['fps'] for m in latency_models]
            
            x = np.arange(len(latency_models))
            width = 0.35
            
            bars1 = ax4.bar(x - width/2, avg_times, width, label='平均推理时间 (秒)', color='skyblue')
            ax4.set_xlabel('模型')
            ax4.set_ylabel('时间 (秒)', color='blue')
            ax4.set_xticks(x)
            ax4.set_xticklabels(latency_models, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # 添加第二个Y轴显示FPS
            ax4_fps = ax4.twinx()
            bars2 = ax4_fps.bar(x + width/2, fps_values, width, label='FPS (帧/秒)', color='lightcoral')
            ax4_fps.set_ylabel('FPS', color='red')
            
            # 添加数值标签
            for bars, values in [(bars1, avg_times), (bars2, fps_values)]:
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2, height + 0.001,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            # 合并图例
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_fps.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
        # 创建数据表格
        if epoch == max_epochs:  # 只在最后一帧显示完整表格
            table_data = []
            for model_name, df in training_results.items():
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    precision = last_row['metrics/precision(B)'] if 'metrics/precision(B)' in df.columns else 'N/A'
                    recall = last_row['metrics/recall(B)'] if 'metrics/recall(B)' in df.columns else 'N/A'
                    map50 = last_row['metrics/mAP50(B)'] if 'metrics/mAP50(B)' in df.columns else 'N/A'
                    
                    if isinstance(precision, (int, float)):
                        precision = f'{precision:.3f}'
                    if isinstance(recall, (int, float)):
                        recall = f'{recall:.3f}'
                    if isinstance(map50, (int, float)):
                        map50 = f'{map50:.3f}'
                    
                    table_data.append([model_name, precision, recall, map50])
            
            # 创建表格
            if table_data:
                col_labels = ['模型', '精确率', '召回率', 'mAP50']
                table = ax5.table(cellText=table_data, colLabels=col_labels,
                                 loc='center', cellLoc='center', colWidths=[0.2, 0.2, 0.2, 0.2])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
                ax5.axis('off')
                
                # 设置表格样式
                for (i, j), cell in table.get_celld().items():
                    if i == 0:  # 标题行
                        cell.set_text_props(fontweight='bold', color='white')
                        cell.set_facecolor('#4CAF50')
                    else:
                        if i % 2 == 0:
                            cell.set_facecolor('#f0f0f0')
        
        # 添加整体信息文本
        info_text = f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n总帧数: {max_epochs}'
        fig.text(0.02, 0.02, info_text, fontsize=9, alpha=0.7)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 将图形转换为图像帧
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(frame)
        
        if epoch % 5 == 0:
            print(f"  已生成 {epoch}/{max_epochs} 帧")
    
    plt.close(fig)
    return frames

def save_gif(frames, output_file, fps=5):
    """保存帧序列为GIF"""
    if not frames:
        print("没有可保存的帧")
        return False
    
    print(f"正在保存GIF: {output_file} ({len(frames)}帧, {fps} FPS)")
    
    # 转换帧为PIL图像
    pil_images = []
    for i, frame in enumerate(frames):
        pil_image = Image.fromarray(frame)
        pil_images.append(pil_image)
    
    # 保存GIF
    pil_images[0].save(output_file, save_all=True, append_images=pil_images[1:],
                      duration=int(1000/fps), loop=0, optimize=True)
    
    print(f"GIF已保存: {output_file} ({os.path.getsize(output_file)/1024/1024:.2f} MB)")
    return True

def main():
    print("=" * 60)
    print("生成训练指标和延迟数据GIF动画")
    print("=" * 60)
    
    # 加载训练结果
    print("1. 加载训练结果数据...")
    training_results = load_training_results()
    if not training_results:
        print("错误: 未找到训练结果文件")
        return
    
    # 获取延迟数据
    print("\n2. 获取模型延迟数据...")
    if RERUN_LATENCY_TEST:
        latency_results = run_latency_test()
    else:
        # 尝试从现有文件加载延迟数据（简化处理）
        latency_results = {}
        print("  使用现有延迟数据（如需最新数据请设置RERUN_LATENCY_TEST=True）")
    
    if not latency_results:
        print("  警告: 未获取到延迟数据，延迟对比图将为空")
    
    # 生成动画帧
    print("\n3. 生成动画帧...")
    frames = create_animation_frames(training_results, latency_results)
    
    if not frames:
        print("错误: 未生成任何动画帧")
        return
    
    # 保存GIF
    print("\n4. 保存GIF动画...")
    success = save_gif(frames, OUTPUT_GIF, GIF_FPS)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ GIF动画生成完成!")
        print(f"   文件: {OUTPUT_GIF}")
        print(f"   尺寸: {len(frames)}帧, {GIF_FPS} FPS")
        print(f"   内容: 训练指标 + 延迟数据动态展示")
        print("=" * 60)
    else:
        print("\n❌ GIF生成失败")

# 导入PIL Image（延迟导入以避免依赖问题）
try:
    from PIL import Image
except ImportError:
    print("正在安装Pillow...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'pillow', '-q'])
    from PIL import Image

if __name__ == "__main__":
    main()