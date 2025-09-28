@echo off
echo ========================================
echo    Urdu to Roman Translation System
echo    Starting Streamlit Web Application
echo ========================================
echo.

cd /d "%~dp0"
echo Current directory: %CD%
echo.

echo Starting Streamlit server...
C:/Users/Hamxa/AppData/Local/Programs/Python/Python313/python.exe -m streamlit run streamlit_app_working.py --server.port 8505

echo.
echo Streamlit app stopped.
pause