@echo off
echo ============================================
echo   Ticket Categorizer - Live Demo
echo ============================================
echo.
echo Installing required packages (pandas, scikit-learn)...
echo This may take a minute the first time.
echo.

python -m pip install --quiet pandas scikit-learn

if %errorlevel% neq 0 (
    echo.
    echo Something went wrong installing packages.
    echo Make sure Python is installed and added to PATH.
    pause
    exit /b 1
)

echo.
echo Setup complete. Starting live demo...
echo Type a ticket subject and body when prompted.
echo Type "quit" at any prompt to exit.
echo.

python ticket_categorizer.py --demo

echo.
pause
