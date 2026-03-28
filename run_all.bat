@echo off
setlocal
cd /d "%~dp0."

set "PYCMD="
where python >nul 2>&1 && set "PYCMD=python"
if not defined PYCMD where py >nul 2>&1 && set "PYCMD=py -3"

if not defined PYCMD (
    echo Python was not found. Install Python 3 and add it to PATH, or install the py launcher.
    pause
    exit /b 1
)

%PYCMD% run_pipeline.py %*
set "_EC=%ERRORLEVEL%"

echo.
if %_EC% neq 0 echo Pipeline exited with code %_EC%.
pause
exit /b %_EC%
