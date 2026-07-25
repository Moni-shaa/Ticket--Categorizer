@echo off
echo ============================================
echo   Ticket Categorizer - Setup and Run
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
echo Setup complete. Running the categorizer...
echo.

python ticket_categorizer.py

echo.
echo ============================================
echo   Done. Press any key to close this window.
echo ============================================
pause >nul
