
import torch
from ultralytics import YOLO

# 修复兼容性
original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = safe_torch_load

# 加载模型
model = YOLO('YOLOv8-app-master/weights/best.pt')

# 简单训练
results = model.train(
    data='quick_train_config.yaml',
    epochs=30,
    imgsz=640,
    batch=8,
    name='quick_apple_train',
    save=True,
    device='cpu',
    lr0=0.001,
    optimizer='Adam',
    augment=False,  # 关闭复杂增强
    plots=False,    # 关闭图表生成
    val=False,      # 跳过验证
)

print("训练完成！")
