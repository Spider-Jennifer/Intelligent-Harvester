"""
智能采收车 - 苹果检测系统
YOLOv8 苹果检测 Streamlit Web 应用
"""
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# 页面配置
st.set_page_config(
    page_title="智能采收车 - 苹果检测系统",
    page_icon="🍎",
    layout="wide"
)

# 标题
st.title("🍎 智能采收车 - 苹果检测系统")
st.markdown("基于 **YOLOv8** 的苹果目标检测系统，支持图片上传和实时检测")

# 侧边栏：模型选择
with st.sidebar:
    st.header("⚙️ 模型配置")

    # 模型选择
    model_option = st.selectbox(
        "选择模型",
        options=["yolov8m.pt (预训练模型)", "apple_detection_model2 (苹果专用模型)"],
        help="yolov8m.pt 为通用预训练模型；专用模型需要先训练"
    )

    if "预训练" in model_option:
        model_path = os.path.join(os.path.dirname(__file__), "legacy", "yolov8-apple-detection-master", "yolov8m.pt")
    else:
        model_path = os.path.join(os.path.dirname(__file__), "legacy", "yolov8-apple-detection-master", "apple_detection_model2", "weights", "best.pt")

    # 置信度滑块
    confidence = st.slider("检测置信度阈值", min_value=0.1, max_value=0.9, value=0.25, step=0.05)

    st.divider()
    st.header("📊 检测信息")
    info_placeholder = st.empty()

# 加载模型
@st.cache_resource
def load_model(path):
    """加载 YOLO 模型"""
    if not os.path.exists(path):
        return None, "模型文件未找到"
    try:
        model = YOLO(path)
        return model, "模型加载成功"
    except Exception as e:
        return None, str(e)

# 执行检测
def detect_objects(model, image, conf=0.25):
    """执行目标检测并返回标注后的图像和结果"""
    results = model(image, conf=conf)
    annotated = results[0].plot()

    # 提取检测信息
    detections = []
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            detections.append({
                "类别": model.names[int(box.cls[0].item())],
                "置信度": f"{box.conf[0].item():.2%}",
                "位置": f"({int(box.xyxy[0][0])}, {int(box.xyxy[0][1])}) - ({int(box.xyxy[0][2])}, {int(box.xyxy[0][3])})"
            })

    return annotated, detections, len(detections)

# 主界面
model, status_msg = load_model(model_path)

if model is None:
    st.error(f"❌ {status_msg}")
    st.info("请先运行训练脚本生成专用模型，或确保 yolov8m.pt 存在于项目中。")
else:
    st.success(f"✅ {status_msg} — 可检测 **{len(model.names)}** 种类别")

    # 显示 COCO 类别中与水果相关的
    fruit_classes = {k: v for k, v in model.names.items() if any(fruit in v.lower() for fruit in ['apple', 'orange', 'banana', 'fruit'])}
    if fruit_classes:
        with st.expander("🍎 水果相关检测类别"):
            cols = st.columns(4)
            for i, (cls_id, cls_name) in enumerate(fruit_classes.items()):
                cols[i % 4].markdown(f"- **{cls_name}** `ID:{cls_id}`")

    # 两个选项卡：上传图片 / 摄像头拍照
    tab1, tab2 = st.tabs(["📁 上传图片", "📷 摄像头拍照"])

    with tab1:
        uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png", "bmp"])

        if uploaded_file is not None:
            # 显示原图和检测结果
            col1, col2 = st.columns(2)

            # 读取图片
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with col1:
                st.subheader("📷 原始图片")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

            # 执行检测
            with st.spinner("🔍 正在检测苹果..."):
                annotated_img, detections, count = detect_objects(model, image, confidence)

            with col2:
                st.subheader(f"🎯 检测结果 (共 {count} 个目标)")
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            # 检测详情
            if detections:
                st.subheader("📋 检测详情")
                # 只显示苹果相关的结果
                apple_detections = [d for d in detections if d["类别"] == "apple"]
                if apple_detections:
                    st.metric("🍎 检测到的苹果数量", len(apple_detections))
                    st.dataframe(apple_detections, use_container_width=True)
                else:
                    st.info("未检测到苹果，但检测到以下目标：")
                    st.dataframe(detections, use_container_width=True)
            else:
                st.info("未检测到任何目标，请尝试调整置信度阈值。")

            # 更新侧边栏信息
            with info_placeholder:
                st.metric("检测目标数", count)
                apple_count = sum(1 for d in detections if d["类别"] == "apple")
                if apple_count > 0:
                    st.metric("🍎 苹果数量", apple_count)

    with tab2:
        st.subheader("📷 摄像头拍照检测")
        camera_image = st.camera_input("点击拍照进行苹果检测")

        if camera_image is not None:
            col1, col2 = st.columns(2)

            file_bytes = np.asarray(bytearray(camera_image.getvalue()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with col1:
                st.subheader("📷 拍摄的照片")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

            with st.spinner("🔍 正在检测苹果..."):
                annotated_img, detections, count = detect_objects(model, image, confidence)

            with col2:
                st.subheader(f"🎯 检测结果 (共 {count} 个目标)")
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            if detections:
                apple_detections = [d for d in detections if d["类别"] == "apple"]
                if apple_detections:
                    st.metric("🍎 检测到的苹果数量", len(apple_detections))
                    st.dataframe(apple_detections, use_container_width=True)

# 底部
st.divider()
st.caption("🍎 智能采收车 | 基于 YOLOv8 | 苹果目标检测系统")
