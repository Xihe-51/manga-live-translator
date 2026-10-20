@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

title 漫画实时翻译 - 启动中

rem Prefer the interpreter that already has the deps installed.
rem Order: conda env, project venv, .venv, then PATH python.
set "PY="

for %%P in ("D:\conda\python.exe" "%~dp0venv\Scripts\python.exe" "%~dp0.venv\Scripts\python.exe") do (
  if not defined PY (
    if exist %%P (
      %%P -c "import cv2, PySide6" >nul 2>nul
      if not errorlevel 1 set "PY=%%~P"
    )
  )
)

if not defined PY (
  where python >nul 2>nul
  if not errorlevel 1 (
    python -c "import cv2, PySide6" >nul 2>nul
    if not errorlevel 1 set "PY=python"
  )
)

if not defined PY (
  echo.
  echo   [错误] 没找到装好依赖的 Python 环境。
  echo   该环境需要同时有 opencv-python 和 PySide6。
  echo   pip install -r requirements.txt
  echo   pip install PySide6
  echo.
  pause
  exit /b 1
)

echo   使用解释器: %PY%
echo   正在启动图形界面...
echo.

"%PY%" main.py --gui %*

rem Keep the window open so errors stay visible.
echo.
echo   程序已退出（退出码 %errorlevel%）。
pause
