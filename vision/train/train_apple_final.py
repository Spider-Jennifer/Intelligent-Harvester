"""
最终版苹果检测模型训练脚本
使用328张苹果照片进行高质量训练
"""

from ultralytics import YOLO
import os

def train_apple_model():
    """训练苹果检测模型"""
    print("开始训练高质量苹果检测模型...")
    print("=" * 50)
    
    # 检查数据集配置
    data_yaml = "YOLO-v8-app/dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"错误: 找不到配置文件 {data_yaml}")
        return
    
    print(f"使用配置文件: {data_yaml}")
    
    # 检查数据集图片数量
    train_dir = "YOLO-v8-app/dataset/images/train"
    val_dir = "YOLO-v8-app/dataset/images/val"
    
    train_count = len([f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    val_count = len([f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"训练集图片: {train_count} 张")
    print(f"验证集图片: {val_count} 张")
    
    if train_count < 50:
        print("警告: 训练集图片数量较少，建议至少50张")
    
    # 训练参数 - 针对高质量苹果检测优化
    training_args = {
        'data': data_yaml,
        'epochs': 50,           # 减少epoch，防止过拟合
        'imgsz': 640,           # 高分辨率
        'batch': 4,             # 小批次，适合CPU
        'workers': 0,           # Windows上设为0避免问题
        'device': 'cpu',        # 使用CPU
        'name': 'apple_final',
        'patience': 10,         # 早停
        'save': True,
        'pretrained': True,     # 使用预训练权重
        'optimizer': 'Adam',    # Adam优化器
        'lr0': 0.001,           # 学习率
        'lrf': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'hsv_h': 0.015,         # 颜色增强
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'fliplr': 0.5,          # 水平翻转
        'degrees': 10.0,        # 旋转增强
        'translate': 0.1,       # 平移增强
        'scale': 0.2,           # 缩放增强
        'shear': 2.0,           # 剪切增强
        'perspective': 0.0005,  # 透视变换
    }
    
    print("\n训练配置:")
    print(f"  Epochs: {training_args['epochs']}")
    print(f"  图像尺寸: {training_args['imgsz']}")
    print(f"  批次大小: {training_args['batch']}")
    print(f"  优化器: {training_args['optimizer']}")
    print(f"  学习率: {training_args['lr0']}")
    
    print("\n开始训练...")
    print("注意: 训练可能需要30-60分钟，请耐心等待")
    print("进度条会显示训练状态")
    
    # 加载预训练模型
    model = YOLO('yolov8n.pt')
    
    # 开始训练
    results = model.train(**training_args)
    
    print("\n训练完成！")
    print("=" * 50)
    
    # 找到最佳模型路径
    best_model_path = "runs/detect/apple_final/weights/best.pt"
    if os.path.exists(best_model_path):
        print(f"最佳模型已保存到: {best_model_path}")
        
        # 复制到项目根目录方便使用
        import shutil
        shutil.copy2(best_model_path, "apple_final.pt")
        print(f"已复制到: apple_final.pt")
        
        # 测试新模型
        print("\n测试新模型...")
        test_model = YOLO("apple_final.pt")
        
        # 测试一张训练图片
        test_image = os.path.join(train_dir, os.listdir(train_dir)[0])
        if os.path.exists(test_image):
            print(f"测试图片: {os.path.basename(test_image)}")
            test_results = test_model(test_image, conf=0.1, verbose=False)  # 降低置信度阈值以提高检测灵敏度
            
            if test_results and len(test_results) > 0:
                result = test_results[0]
                if result.boxes is not None:
                    print(f"检测到 {len(result.boxes)} 个苹果")
                else:
                    print("未检测到苹果")
    else:
        print("\n警告: 找不到最佳模型文件")
        print("请检查 runs/detect/apple_final/ 目录")
    
    print("\n下一步:")
    print("1. 测试模型: python test_trained_model.py apple_final.pt")
    print("2. 更新应用中的模型路径")
    print("3. 运行应用测试效果")
    print("=" * 50)

if __name__ == "__main__":
    train_apple_model()