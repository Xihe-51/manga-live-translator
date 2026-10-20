@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

title 漫画实时翻译 - 运行测试

rem Prefer the interpreter that already has the deps installed.
set "PY="

for %%P in ("D:\conda\python.exe" "%~dp0venv\Scripts\python.exe" "%~dp0.venv\Scripts\python.exe") do (
  if not defined PY (
    if exist %%P (
      %%P -c "import cv2" >nul 2>nul
      if not errorlevel 1 set "PY=%%~P"
    )
  )
)

if not defined PY (
  where python >nul 2>nul
  if not errorlevel 1 (
    python -c "import cv2" >nul 2>nul
    if not errorlevel 1 set "PY=python"
  )
)

if not defined PY (
  echo   [错误] 没找到装好 opencv-python 的 Python 环境。
  pause
  exit /b 1
)

echo   使用解释器: %PY%
echo   正在运行测试...
echo.
"%PY%" -m pytest tests -q

echo.
echo   测试结束（退出码 %errorlevel%）。
pause
