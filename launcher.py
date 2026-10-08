"""
π-因果沙盒 · 桌面启动器
双击后自动启动 Streamlit 并打开浏览器。
"""

import subprocess
import webbrowser
import time
import sys
import os
import socket


def find_free_port(start=8501, end=8600):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise RuntimeError("8501-8600 端口都被占用")


def main():
    # 切到脚本所在目录（关键，否则找不到 chat_llm.py）
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    port = find_free_port()
    print("=" * 60)
    print("  π-因果沙盒 · LLM 对话版")
    print("=" * 60)
    print(f"\n正在启动，端口 {port} ...")
    print("浏览器会自动打开。")
    print("\n⚠️ 关闭这个窗口 = 关闭程序。\n")

    # 启动 streamlit
    proc = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "chat_llm.py",
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ])

    # 等 4 秒让 streamlit 起来，然后打开浏览器
    time.sleep(4)
    webbrowser.open(f"http://localhost:{port}")

    # 等 streamlit 结束（用户关窗口时）
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()