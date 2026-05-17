#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试Web应用连接问题
"""

import socket
import requests
import subprocess
import sys
import time

def test_port(port):
    """测试端口是否可访问"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        return result == 0
    except:
        return False

def test_http(port):
    """测试HTTP连接"""
    try:
        response = requests.get(f'http://localhost:{port}', timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("=" * 60)
    print("🔍 Web应用连接问题诊断")
    print("=" * 60)
    
    ports_to_test = [8600, 8520, 8501, 8503]
    
    print("\n📊 端口状态检查:")
    for port in ports_to_test:
        port_open = test_port(port)
        http_ok = test_http(port) if port_open else False
        status = "✅ 运行中" if http_ok else "🟡 端口开放但HTTP失败" if port_open else "❌ 未运行"
        print(f"   端口 {port}: {status}")
    
    print("\n🔧 解决方案:")
    print("1. 如果端口未运行:")
    print("   - 双击 run_fast_camera.bat (端口8600)")
    print("   - 双击 run_project_now.bat (端口8520)")
    
    print("\n2. 如果端口运行但无法访问:")
    print("   - 检查防火墙设置")
    print("   - 尝试使用 127.0.0.1 代替 localhost")
    print("   - 尝试不同的浏览器")
    
    print("\n3. 快速测试:")
    print("   - http://localhost:8520 (原始应用)")
    print("   - http://localhost:8600 (优化应用)")
    print("   - http://127.0.0.1:8520 (使用IP地址)")
    
    print("\n4. 重启应用:")
    print("   - 按 Ctrl+C 停止当前应用")
    print("   - 重新运行批处理文件")
    
    print("\n🚀 立即启动优化版Web应用:")
    choice = input("   是否启动优化版Web应用? (y/n): ")
    if choice.lower() == 'y':
        print("\n正在启动优化版Web应用...")
        try:
            subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "fast_camera_app.py", 
                "--server.port", "8600",
                "--server.headless", "true"
            ])
            print("✅ 应用已启动，请访问: http://localhost:8600")
            time.sleep(2)
            import webbrowser
            webbrowser.open("http://localhost:8600")
        except Exception as e:
            print(f"❌ 启动失败: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()