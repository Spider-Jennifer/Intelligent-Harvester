#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
苹果摄像头检测 - 核心功能版
"""

import streamlit as st
import cv2
import numpy as np
import time
import os

# 页面配置
st.set_page_config(page_title="苹果检测", layout="wide")

# 标题
st.title("🍎 苹果摄像头检测")

# 侧边栏配置
st.sidebar.header("检测设置")

# 模型选择
model_options = {
    "yolov8n (预训练)": "yolov8n.pt",
    "apple_best (之前训练)": "apple_best.pt",
    "apple_quick (新训练)": "apple_quick.pt",
    "apple_sensitive (高灵敏度-推荐)": "apple_sensitive.pt"
}
selected_model = st.sidebar.selectbox("选择模型", list(model_options.keys()), index=3)

# 置信度
confidence = st.sidebar.slider("置信度", 0.1, 0.9, 0.1, 0.05)  # 默认值改为0.1以提高检测灵敏度

# 分辨率
resolution = st.sidebar.selectbox("分辨率", ["320x240", "640x480"], index=1)
width, height = map(int, resolution.split('x'))

# 初始化状态
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
    st.session_state.frame_count = 0

# 加载模型
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model_path = model_options[selected_model]
        if not os.path.exists(model_path):
            model_path = "yolov8n.pt"
        return YOLO(model_path)
    except:
        return None

model = load_model()

def detect_frame(frame):
    """检测单帧"""
    if model is None:
        return frame, 0
    
    try:
        # 使用与训练相同的推理尺寸（正方形）
        inference_size = 320
        resized = cv2.resize(frame, (inference_size, inference_size))
        results = model(resized, conf=confidence, verbose=False)
        
        detections = 0
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                # 缩放比例 - 保持正方形比例
                scale_x = frame.shape[1] / inference_size
                scale_y = frame.shape[0] / inference_size
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    conf = confidences[i]
                    
                    # 只显示置信度大于阈值的检测
                    if conf < confidence:
                        continue
                    
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    # 绘制边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 添加标签：apple + 置信度
                    label = f"apple {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # 标签背景
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), 
                                 (x1 + label_size[0], y1), (0, 255, 0), -1)
                    
                    # 标签文字
                    cv2.putText(frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    detections += 1
        
        return frame, detections
    except:
        return frame, 0

# 主界面
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("实时画面")
    video_placeholder = st.empty()
    
    # 控制按钮
    if not st.session_state.camera_running:
        if st.button("启动摄像头", type="primary"):
            st.session_state.camera_running = True
            st.session_state.frame_count = 0
            st.rerun()
    else:
        if st.button("停止摄像头"):
            st.session_state.camera_running = False
            st.rerun()

with col2:
    st.subheader("检测信息")
    info_placeholder = st.empty()

# 摄像头循环
if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.error("摄像头读取失败")
                break
            
            # 转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 检测
            processed_frame, detections = detect_frame(frame_rgb.copy())
            
            # 显示画面
            video_placeholder.image(processed_frame, 
                                   caption=f"检测到 {detections} 个苹果")
            
            # 更新信息
            st.session_state.frame_count += 1
            info_placeholder.markdown(f"""
            **实时统计：**
            - 处理帧数: {st.session_state.frame_count}
            - 检测数量: {detections}
            - 置信度: {confidence}
            - 分辨率: {resolution}
            """)
            
            # 控制帧率
            time.sleep(0.05)
        
        cap.release()
    else:
        st.error("无法打开摄像头")
else:
    st.info("点击'启动摄像头'开始检测")