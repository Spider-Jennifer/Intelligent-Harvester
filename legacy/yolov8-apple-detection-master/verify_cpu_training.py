import torch
import sys

# 检查PyTorch版本和设备可用性
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"CUDA设备数量: {torch.cuda.device_count()}")
print(f"当前设备: CPU")

# 尝试创建一个简单的模型来测试CPU训练
print("\n测试CPU训练功能:")
try:
    # 创建一个简单的模型
    model = torch.nn.Linear(10, 1)
    
    # 创建一些随机数据
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    
    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 进行一次前向传播和反向传播
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    print("✅ CPU训练测试成功!")
    print(f"损失值: {loss.item():.4f}")
    print("CPU训练功能正常，可以在GPU版本安装前使用CPU进行训练。")
except Exception as e:
    print(f"❌ CPU训练测试失败: {str(e)}")
    print("请检查PyTorch安装是否正确。")

print("\n提示: 当您完成PyTorch GPU版本安装后，我们可以运行check_cuda.py脚本来验证GPU是否可用。")