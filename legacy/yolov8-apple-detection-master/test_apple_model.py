import os
import argparse
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

# 设置基础目录为脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8苹果检测模型测试')
    parser.add_argument('--model', type=str, 
                        default=os.path.join(BASE_DIR, 'apple_detection_model2', 'weights', 'best.pt'),
                        help='模型文件路径')
    parser.add_argument('--test_dir', type=str,
                        default=os.path.join(BASE_DIR, 'Dataset', 'apple_dataset', 'images', 'train'),
                        help='测试图像目录')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(BASE_DIR, 'test_results'),
                        help='结果保存目录')
    parser.add_argument('--num_images', type=int, default=5, help='测试图像数量')
    parser.add_argument('--conf', type=float, default=0.25, help='检测置信度阈值')
    return parser.parse_args()

# 先创建默认输出目录
default_output_dir = os.path.join(BASE_DIR, 'test_results')
os.makedirs(default_output_dir, exist_ok=True)

def load_and_test_model():
    args = parse_args()
    
    print(f"尝试加载模型：{args.model}")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print("错误：模型文件不存在！训练可能还未完成或保存路径不正确。")
        print("请等待训练完成后再运行此测试脚本。")
        print("或者使用 --model 参数指定正确的模型路径")
        
        # 尝试查找其他可能的模型文件
        possible_models = []
        for root, _, files in os.walk(BASE_DIR):
            for file in files:
                if file.endswith('.pt') and ('best' in file or 'last' in file):
                    possible_models.append(os.path.join(root, file))
        
        if possible_models:
            print("\n在项目中找到以下模型文件：")
            for i, model_path in enumerate(possible_models, 1):
                rel_path = os.path.relpath(model_path, BASE_DIR)
                print(f"{i}. {rel_path}")
            print("\n请使用 --model 参数指定要使用的模型，例如：")
            print(f"python test_apple_model.py --model {os.path.relpath(possible_models[0], BASE_DIR)}")
        
        return False
    
    try:
        # 加载模型
        model = YOLO(args.model)
        print("模型加载成功！")
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 获取测试图像列表
        image_files = [f for f in os.listdir(args.test_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"未找到测试图像在目录：{args.test_dir}")
            return False
        
        # 选择指定数量的图像进行测试
        test_images = image_files[:args.num_images]
        print(f"找到了{len(image_files)}张图像，将测试前{len(test_images)}张")
        
        # 对每张图像进行预测
        for image_file in test_images:
            image_path = os.path.join(args.test_dir, image_file)
            print(f"处理图像：{image_file}")
            
            # 执行预测，设置置信度阈值
            results = model(image_path, conf=args.conf)
            
            # 保存结果图像
            for i, r in enumerate(results):
                # 绘制边界框和标签
                im_array = r.plot()  # 绘制结果到图像
                
                # 保存结果
                output_path = os.path.join(args.output_dir, f"result_{image_file}")
                cv2.imwrite(output_path, im_array)
                print(f"结果已保存到：{output_path}")
                
                # 打印检测到的物体信息
                boxes = r.boxes
                print(f"在图像中检测到 {len(boxes)} 个物体")
                for box in boxes:
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    class_name = model.names[class_id]
                    print(f"  - {class_name}: 置信度 = {confidence:.2f}")
        
        print("\n测试完成！所有结果已保存到：", args.output_dir)
        return True
        
    except Exception as e:
        print(f"测试过程中出错：{e}")
        return False

def main():
    print("======== 苹果检测模型测试 ========")
    success = load_and_test_model()
    
    if success:
        print("\n模型测试成功！您可以通过运行应用程序来使用这个模型：")
        print("1. 进入 YOLO-v8-app/YOLOv8-app-master 目录")
        print("2. 运行命令：streamlit run app.py")
        print("3. 在界面中，选择 'apple_detection_model' 作为检测模型")
        print("4. 上传包含苹果的图像进行检测")
    else:
        print("\n模型测试失败。请检查模型路径或等待训练完成。")

if __name__ == "__main__":
    main()