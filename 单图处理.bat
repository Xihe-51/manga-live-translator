@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

title 漫画实时翻译 - 单图处理

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

set "IMG=%~1"
if "%IMG%"=="" set "IMG=examples\test.png"

echo   使用解释器: %PY%
echo   输入: %IMG%
echo   输出: output.png
echo.
"%PY%" main.py --image "%IMG%" --out output.png

echo.
echo   处理结束（退出码 %errorlevel%）。
pause
