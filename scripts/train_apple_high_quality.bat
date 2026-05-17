@echo off
chcp 65001 >nul
echo ========================================
echo   开始训练高精度苹果检测模型
echo ========================================
echo.

echo 检查数据集配置...
python -c "
import os

data_yaml = 'YOLO-v8-app/dataset/data.yaml'
if os.path.exists(data_yaml):
    print(f'✓ 找到配置文件: {data_yaml}')
    
    # 检查图片数量
    train_dir = 'YOLO-v8-app/dataset/images/train'
    val_dir = 'YOLO-v8-app/dataset/images/val'
    
    train_count = len([f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    val_count = len([f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f'  训练集图片: {train_count} 张')
    print(f'  验证集图片: {val_count} 张')
    
    if train_count >= 50:
        print('✓ 数据集准备就绪，可以开始训练')
    else:
        print('⚠ 警告: 训练集图片数量较少，建议至少50张')
else:
    print('✗ 错误: 找不到配置文件')
    print('请先运行 prepare_and_train.bat 准备数据集')
"
echo.
pause

echo.
echo 开始训练高质量苹果检测模型...
echo 注意: 训练可能需要45-90分钟
echo 请保持终端打开，不要关闭
echo.
echo 按任意键开始训练...
pause >nul

python train_apple_high_quality.py

echo.
echo ========================================
echo   训练完成！
echo ========================================
echo.
echo 下一步:
echo 1. 测试新模型: python test_trained_model.py apple_high_quality.pt
echo 2. 更新应用中的模型路径
echo 3. 运行应用测试效果
echo.
pause