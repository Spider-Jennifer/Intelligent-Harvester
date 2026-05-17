#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
立即运行YOLOv8苹果检测项目
"""

import subprocess
import sys
import time
import webbrowser
import os

def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """打印标题"""
    print("=" * 60)
    print("🚀 YOLOv8 苹果检测项目 - 一键启动")
    print("=" * 60)
    print()

def print_options():
    """打印选项"""
    print("请选择运行模式：")
    print()
    print("[1] 优化版Web应用 (推荐)")
    print("    特点：解决摄像头延迟问题，Web界面")
    print("    端口：8600")
    print("    访问：http://localhost:8600")
    print()
    print("[2] 最小延迟版本")
    print("    特点：最低延迟，纯OpenCV界面")
    print("    无Web界面，最高性能")
    print()
    print("[3] 原始Web应用")
    print("    特点：原始项目界面")
    print("    端口：8520")
    print("    访问：http://localhost:8520")
    print()
    print("[4] 退出")
    print()

def run_optimized_web():
    """运行优化版Web应用"""
    print()
    print("🚀 正在启动优化版Web应用...")
    print("端口：8600")
    print("访问地址：http://localhost:8600")
    print("按 Ctrl+C 停止应用")
    print()
    
    # 清理旧进程
    try:
        subprocess.run(["taskkill", "/f", "/im", "python.exe"], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["taskkill", "/f", "/im", "streamlit.exe"], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass
    
    time.sleep(2)
    
    # 在浏览器中打开
    try:
        webbrowser.open("http://localhost:8600")
        print("✅ 已尝试在浏览器中打开应用")
    except:
        print("⚠️  无法自动打开浏览器，请手动访问上述网址")
    
    print()
    print("⏳ 启动Streamlit应用中...")
    print("-" * 40)
    
    # 启动Streamlit应用
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "fast_camera_app.py", 
            "--server.port", "8600",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

def run_minimal_latency():
    """运行最小延迟版本"""
    print()
    print("🚀 正在启动最小延迟版本...")
    print("特点：纯OpenCV，最低延迟")
    print("控制：")
    print("  - Q 或 ESC: 退出")
    print("  - C: 切换置信度 (0.25/0.5/0.75)")
    print("  - S: 切换跳帧设置 (1/2/3)")
    print("  - R: 重置性能统计")
    print("  - +: 增加置信度")
    print("  - -: 降低置信度")
    print()
    
    time.sleep(2)
    
    try:
        subprocess.run([sys.executable, "minimal_camera.py"])
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

def main():
    clear_screen()
    print_header()
    
    while True:
        print_options()
        
        try:
            choice = input("请选择 (1-4): ").strip()
            
            if choice == "1":
                run_optimized_web()
                break
            elif choice == "2":
                run_minimal_latency()
                break
            elif choice == "3":
                print("\n📝 原始Web应用位于: YOLO-v8-app\\YOLOv8-app-master\\")
                print("   请双击运行: run_project_now.bat")
                print("   或访问: http://localhost:8520")
                break
            elif choice == "4":
                print("\n👋 再见！")
                break
            else:
                print("\n❌ 无效选择，请重新输入")
                time.sleep(1)
                clear_screen()
                print_header()
                
        except KeyboardInterrupt:
            print("\n\n👋 程序已取消")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            break
    
    print()
    input("按 Enter 键退出...")

if __name__ == "__main__":
    main()