try:
    import streamlit
    print("streamlit installed:", streamlit.__version__)
except ImportError:
    print("streamlit not installed")

try:
    import ultralytics
    print("ultralytics installed:", ultralytics.__version__)
except ImportError:
    print("ultralytics not installed")

try:
    import torch
    print("torch installed:", torch.__version__)
except ImportError:
    print("torch not installed")

try:
    import cv2
    print("opencv installed:", cv2.__version__)
except ImportError:
    print("opencv not installed")