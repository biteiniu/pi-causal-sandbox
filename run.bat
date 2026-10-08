@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo   pi-causal-sandbox
echo ============================================================
echo.
echo 浏览器将自动打开，请稍等 6 秒...
echo 黑框请勿关闭（关闭 = 退出程序）
echo.

start "" /b cmd /c "timeout /t 6 /nobreak >nul && start http://127.0.0.1:8501"

D:\anaconda3\python.exe -m streamlit run chat_llm.py --server.headless=true --server.port=8501

pause