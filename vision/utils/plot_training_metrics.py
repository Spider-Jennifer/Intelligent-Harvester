#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制训练指标图表 - 识别率可视化
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_training_charts():
    """创建训练指标图表"""
    
    print("=" * 60)
    print("Training Metrics Visualization")
    print("=" * 60)
    
    # 检查结果文件
    results_file = "runs/cpu_long_training/apple_cpu_long/results.csv"
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        print("Please wait for training to generate results")
        return
    
    print(f"Reading results from: {results_file}")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(results_file)
        
        print(f"Total epochs recorded: {len(df)}")
        print(f"Available columns: {list(df.columns)}")
        
        # 检查必要的列
        required_columns = ['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            print("Available columns:", list(df.columns))
            
            # 尝试查找相似的列名
            for col in df.columns:
                if 'mAP' in col or 'map' in col:
                    print(f"  Found similar: {col}")
        
        # 创建图表
        plt.figure(figsize=(15, 10))
        
        # 1. mAP50 和 mAP50-95 图表
        plt.subplot(2, 2, 1)
        if 'metrics/mAP50(B)' in df.columns:
            plt.plot(df['epoch'], df['metrics/mAP50(B)'], 'b-', linewidth=2, label='mAP50')
        if 'metrics/mAP50-95(B)' in df.columns:
            plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], 'r-', linewidth=2, label='mAP50-95')
        
        plt.xlabel('Epoch')
        plt.ylabel('mAP Score')
        plt.title('Detection Accuracy (mAP)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2. 精确率和召回率
        plt.subplot(2, 2, 2)
        if 'metrics/precision(B)' in df.columns:
            plt.plot(df['epoch'], df['metrics/precision(B)'], 'g-', linewidth=2, label='Precision')
        if 'metrics/recall(B)' in df.columns:
            plt.plot(df['epoch'], df['metrics/recall(B)'], 'm-', linewidth=2, label='Recall')
        
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Precision and Recall')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 3. 损失函数
        plt.subplot(2, 2, 3)
        if 'train/box_loss' in df.columns:
            plt.plot(df['epoch'], df['train/box_loss'], 'b-', linewidth=1, alpha=0.7, label='Box Loss')
        if 'train/cls_loss' in df.columns:
            plt.plot(df['epoch'], df['train/cls_loss'], 'r-', linewidth=1, alpha=0.7, label='Class Loss')
        if 'train/dfl_loss' in df.columns:
            plt.plot(df['epoch'], df['train/dfl_loss'], 'g-', linewidth=1, alpha=0.7, label='DFL Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 4. 验证损失
        plt.subplot(2, 2, 4)
        if 'val/box_loss' in df.columns:
            plt.plot(df['epoch'], df['val/box_loss'], 'b-', linewidth=2, label='Val Box Loss')
        if 'val/cls_loss' in df.columns:
            plt.plot(df['epoch'], df['val/cls_loss'], 'r-', linewidth=2, label='Val Class Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_metrics_summary.png', dpi=150, bbox_inches='tight')
        print(f"Chart saved as: training_metrics_summary.png")
        
        # 创建单独的识别率图表
        plt.figure(figsize=(12, 8))
        
        # 识别率图表
        ax1 = plt.subplot(2, 1, 1)
        if 'metrics/mAP50(B)' in df.columns:
            ax1.plot(df['epoch'], df['metrics/mAP50(B)'], 'b-', linewidth=3, label='mAP50 (IoU=0.5)')
        if 'metrics/mAP50-95(B)' in df.columns:
            ax1.plot(df['epoch'], df['metrics/mAP50-95(B)'], 'r-', linewidth=3, label='mAP50-95 (IoU=0.5:0.95)')
        
        ax1.set_xlabel('Training Epoch', fontsize=12)
        ax1.set_ylabel('mAP Score', fontsize=12)
        ax1.set_title('Apple Detection Accuracy Progress', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # 添加当前值标注
        if len(df) > 0:
            last_row = df.iloc[-1]
            if 'metrics/mAP50(B)' in df.columns:
                last_map50 = last_row['metrics/mAP50(B)']
                ax1.annotate(f'{last_map50:.3f}', 
                           xy=(last_row['epoch'], last_map50),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                           fontsize=10)
            
            if 'metrics/mAP50-95(B)' in df.columns:
                last_map = last_row['metrics/mAP50-95(B)']
                ax1.annotate(f'{last_map:.3f}', 
                           xy=(last_row['epoch'], last_map),
                           xytext=(10, -20), textcoords='offset points',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                           fontsize=10)
        
        # 精确率和召回率图表
        ax2 = plt.subplot(2, 1, 2)
        if 'metrics/precision(B)' in df.columns:
            ax2.plot(df['epoch'], df['metrics/precision(B)'], 'g-', linewidth=3, label='Precision')
        if 'metrics/recall(B)' in df.columns:
            ax2.plot(df['epoch'], df['metrics/recall(B)'], 'm-', linewidth=3, label='Recall')
        
        ax2.set_xlabel('Training Epoch', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Precision and Recall Progress', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        # 添加当前值标注
        if len(df) > 0:
            if 'metrics/precision(B)' in df.columns:
                last_precision = last_row['metrics/precision(B)']
                ax2.annotate(f'{last_precision:.3f}', 
                           xy=(last_row['epoch'], last_precision),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                           fontsize=10)
            
            if 'metrics/recall(B)' in df.columns:
                last_recall = last_row['metrics/recall(B)']
                ax2.annotate(f'{last_recall:.3f}', 
                           xy=(last_row['epoch'], last_recall),
                           xytext=(10, -20), textcoords='offset points',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="pink", alpha=0.7),
                           fontsize=10)
        
        plt.tight_layout()
        plt.savefig('detection_accuracy_chart.png', dpi=150, bbox_inches='tight')
        print(f"Accuracy chart saved as: detection_accuracy_chart.png")
        
        # 显示统计信息
        print("\n" + "=" * 60)
        print("Training Statistics Summary")
        print("=" * 60)
        
        if len(df) > 0:
            print(f"Current epoch: {int(last_row['epoch'])}")
            
            if 'metrics/mAP50(B)' in df.columns:
                map50_values = df['metrics/mAP50(B)'].dropna()
                if len(map50_values) > 0:
                    print(f"mAP50: {map50_values.iloc[-1]:.4f} (current)")
                    print(f"mAP50 best: {map50_values.max():.4f} at epoch {map50_values.idxmax() + 1}")
                    print(f"mAP50 improvement: {map50_values.iloc[-1] - map50_values.iloc[0]:+.4f}")
            
            if 'metrics/mAP50-95(B)' in df.columns:
                map_values = df['metrics/mAP50-95(B)'].dropna()
                if len(map_values) > 0:
                    print(f"mAP50-95: {map_values.iloc[-1]:.4f} (current)")
                    print(f"mAP50-95 best: {map_values.max():.4f} at epoch {map_values.idxmax() + 1}")
            
            if 'metrics/precision(B)' in df.columns:
                precision_values = df['metrics/precision(B)'].dropna()
                if len(precision_values) > 0:
                    print(f"Precision: {precision_values.iloc[-1]:.4f} (current)")
            
            if 'metrics/recall(B)' in df.columns:
                recall_values = df['metrics/recall(B)'].dropna()
                if len(recall_values) > 0:
                    print(f"Recall: {recall_values.iloc[-1]:.4f} (current)")
        
        # 创建HTML报告
        create_html_report(df)
        
        print("\n" + "=" * 60)
        print("Charts generated successfully!")
        print("1. training_metrics_summary.png - 综合训练指标")
        print("2. detection_accuracy_chart.png - 识别率详细图表")
        print("3. training_report.html - HTML训练报告")
        print("=" * 60)
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating charts: {e}")
        import traceback
        traceback.print_exc()

def create_html_report(df):
    """创建HTML训练报告"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>YOLOv8 Apple Detection Training Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
            .metric-card { background: white; border-radius: 10px; padding: 20px; 
                          margin: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
                          display: inline-block; width: 200px; text-align: center; }
            .metric-value { font-size: 32px; font-weight: bold; margin: 10px 0; }
            .metric-label { color: #666; font-size: 14px; }
            .chart-container { margin: 30px 0; }
            .improvement { color: green; font-weight: bold; }
            .warning { color: orange; font-weight: bold; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🍎 Apple Detection Model Training Report</h1>
            <p>YOLOv8 Long Training Progress Analysis</p>
        </div>
    """
    
    # 添加统计信息
    if len(df) > 0:
        last_row = df.iloc[-1]
        html_content += """
        <div style="text-align: center;">
            <h2>📊 Current Training Status</h2>
        """
        
        if 'epoch' in df.columns:
            html_content += f"""
            <div class="metric-card">
                <div class="metric-label">Current Epoch</div>
                <div class="metric-value">{int(last_row['epoch'])}</div>
                <div>of 150 total</div>
            </div>
            """
        
        if 'metrics/mAP50(B)' in df.columns:
            map50 = last_row['metrics/mAP50(B)']
            map50_values = df['metrics/mAP50(B)'].dropna()
            improvement = map50_values.iloc[-1] - map50_values.iloc[0] if len(map50_values) > 1 else 0
            
            html_content += f"""
            <div class="metric-card">
                <div class="metric-label">mAP50 Score</div>
                <div class="metric-value">{map50:.3f}</div>
                <div class="improvement">+{improvement:.3f} improvement</div>
            </div>
            """
        
        if 'metrics/mAP50-95(B)' in df.columns:
            map9595 = last_row['metrics/mAP50-95(B)']
            html_content += f"""
            <div class="metric-card">
                <div class="metric-label">mAP50-95 Score</div>
                <div class="metric-value">{map9595:.3f}</div>
                <div>Overall accuracy</div>
            </div>
            """
        
        if 'metrics/precision(B)' in df.columns:
            precision = last_row['metrics/precision(B)']
            html_content += f"""
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{precision:.3f}</div>
                <div>Correct detections</div>
            </div>
            """
        
        if 'metrics/recall(B)' in df.columns:
            recall = last_row['metrics/recall(B)']
            html_content += f"""
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{recall:.3f}</div>
                <div>Objects found</div>
            </div>
            """
        
        html_content += "</div>"
        
        # 添加图表
        html_content += """
        <div class="chart-container">
            <h2>📈 Training Progress Charts</h2>
            <p>Generated charts showing detection accuracy progress:</p>
            <ul>
                <li><strong>training_metrics_summary.png</strong> - Comprehensive training metrics</li>
                <li><strong>detection_accuracy_chart.png</strong> - Detailed accuracy visualization</li>
            </ul>
        </div>
        """
        
        # 添加数据表格
        html_content += """
        <div>
            <h2>📋 Recent Training Metrics</h2>
            <table>
                <tr>
                    <th>Epoch</th>
                    <th>mAP50</th>
                    <th>mAP50-95</th>
                    <th>Precision</th>
                    <th>Recall</th>
                </tr>
        """
        
        # 显示最后10个epoch的数据
        for i in range(min(10, len(df))):
            idx = len(df) - 1 - i
            row = df.iloc[idx]
            
            map50 = row['metrics/mAP50(B)'] if 'metrics/mAP50(B)' in df.columns else 'N/A'
            map9595 = row['metrics/mAP50-95(B)'] if 'metrics/mAP50-95(B)' in df.columns else 'N/A'
            precision = row['metrics/precision(B)'] if 'metrics/precision(B)' in df.columns else 'N/A'
            recall = row['metrics/recall(B)'] if 'metrics/recall(B)' in df.columns else 'N/A'
            
            html_content += f"""
                <tr>
                    <td>{int(row['epoch'])}</td>
                    <td>{map50 if isinstance(map50, str) else f'{map50:.3f}'}</td>
                    <td>{map9595 if isinstance(map9595, str) else f'{map9595:.3f}'}</td>
                    <td>{precision if isinstance(precision, str) else f'{precision:.3f}'}</td>
                    <td>{recall if isinstance(recall, str) else f'{recall:.3f}'}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </div>
        """
    
    html_content += """
        <div style="margin-top: 40px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <h3>📝 Training Information</h3>
            <ul>
                <li><strong>Model:</strong> YOLOv8 Apple Detection</li>
                <li><strong>Training device:</strong> CPU</li>
                <li><strong>Total epochs:</strong> 150</li>
                <li><strong>Batch size:</strong> 8</li>
                <li><strong>Image size:</strong> 640x640</li>
                <li><strong>Dataset:</strong> Apple detection dataset</li>
                <li><strong>Report generated:</strong> """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open('training_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report saved as: training_report.html")

if __name__ == "__main__":
    create_training_charts()