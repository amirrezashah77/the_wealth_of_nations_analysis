@echo off
rem Change directory to the folder where this script is located
cd /d "%~dp0"

echo Activating virtual environment...
rem Assumes your venv folder is named 'venv' and is in the same directory
call "venv\Scripts\activate.bat"

echo Starting Streamlit app...
streamlit run main.py

pause