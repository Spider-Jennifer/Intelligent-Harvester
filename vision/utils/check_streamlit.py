import sys
try:
    import streamlit
    print(f"Streamlit已安装，版本: {streamlit.__version__}")
except ImportError:
    print("Streamlit未安装")
    print("正在安装Streamlit...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "-q"])
    print("Streamlit安装完成")