@echo off
echo ===================================================
echo   KrystalStack / GAMESA Installer Builder
echo ===================================================
echo.
echo This script will package the Python Wizard into a standalone .EXE file.
echo Requirements: Python installed and added to PATH.
echo.

echo 1. Installing PyInstaller...
pip install pyinstaller

echo.
echo 2. Building Executable...
echo    - Runtime Engine
echo    - Guardian Engine
echo    - Solution Inventor
echo.

pyinstaller --onefile --clean --name "Setup_KrystalStack_Wizard" --hidden-import=guardian_hero_engine --hidden-import=solution_inventor --hidden-import=runtime_engine install_wizard.py

echo.
echo ===================================================
echo   BUILD COMPLETE
echo ===================================================
echo.
echo You can find your installer in the 'dist' folder:
echo dist\Setup_KrystalStack_Wizard.exe
echo.
pause
