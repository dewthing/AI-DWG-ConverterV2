@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title AI CAD Converter - PC Setup

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows\setup_windows.ps1"
set "SETUP_EXIT=%ERRORLEVEL%"

if not "%SETUP_EXIT%"=="0" (
  echo.
  echo Setup did not complete. Read the error above, then run SETUP_PC.bat again.
  pause
)

exit /b %SETUP_EXIT%
