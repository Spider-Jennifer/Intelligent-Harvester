import sys

print("检查依赖...")
deps = ["streamlit", "ultralytics", "cv2", "numpy", "PIL"]

for dep in deps:
    try:
        if dep == "cv2":
            import cv2
            print(f"{dep}: 已安装")
        elif dep == "PIL":
            from PIL import Image
            print(f"{dep}: 已安装")
        else:
            __import__(dep)
            print(f"{dep}: 已安装")
    except ImportError:
        print(f"{dep}: 未安装")
        if dep == "cv2":
            dep_name = "opencv-python"
        elif dep == "PIL":
            dep_name = "pillow"
        else:
            dep_name = dep
        print(f"正在安装 {dep_name}...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep_name, "-q"])
        print(f"{dep_name} 安装完成")

print("\n所有依赖检查完成！")