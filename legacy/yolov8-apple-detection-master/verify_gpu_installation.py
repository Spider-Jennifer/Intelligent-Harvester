import torch
import sys

print("===== PyTorch GPU安装验证脚本 =====\n")

# 检查PyTorch版本
print(f"PyTorch版本: {torch.__version__}")

# 检查CUDA可用性
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"CUDA设备数量: {torch.cuda.device_count()}")

# 如果CUDA可用，显示更多信息
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"当前CUDA设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存总容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 测试GPU计算功能
    print("\n测试GPU计算功能:")
    try:
        # 创建在GPU上的张量
        x = torch.randn(10000, 10000).cuda()
        y = torch.randn(10000, 10000).cuda()
        
        # 执行矩阵乘法
        print("执行矩阵乘法...")
        result = torch.matmul(x, y)
        print(f"GPU计算成功! 结果形状: {result.shape}")
        
        # 测试模型训练
        print("\n测试简单模型在GPU上的训练:")
        model = torch.nn.Linear(10, 1).cuda()
        x = torch.randn(5, 10).cuda()
        y = torch.randn(5, 1).cuda()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        print(f"✅ GPU训练测试成功!")
        print(f"损失值: {loss.item():.4f}")
        print("\n🎉 恭喜! PyTorch GPU版本已成功安装并可以正常使用!")
        print("您现在可以使用GPU进行模型训练了。")
        
    except Exception as e:
        print(f"❌ GPU计算测试失败: {str(e)}")
        print("请检查GPU驱动和CUDA安装。")
else:
    print("\n❌ CUDA不可用，请确保已正确安装GPU版本的PyTorch。")
    print("请使用以下命令重新安装:")
    print("pip uninstall -y torch torchvision torchaudio")
    print("pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124 -i https://pypi.tuna.tsinghua.edu.cn/simple")

print("\n安装完成后，请再次运行此脚本来验证GPU是否正常工作。")