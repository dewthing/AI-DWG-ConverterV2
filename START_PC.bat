@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title AI CAD Converter - Local PC

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows\start_windows.ps1"
set "START_EXIT=%ERRORLEVEL%"

if not "%START_EXIT%"=="0" (
  echo.
  echo The program stopped with an error. Read the message above.
  pause
)

exit /b %START_EXIT%
