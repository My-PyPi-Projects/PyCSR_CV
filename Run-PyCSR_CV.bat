@echo off
cd /d "%~dp0"
if exist "dist\PyCSR_CV\PyCSR_CV.exe" (
  start "" "dist\PyCSR_CV\PyCSR_CV.exe"
) else (
  python -m pycsr_cv
)
