@echo off

set "PROJECT=011-Batch-Normalization"

echo Creating project structure...
echo.

mkdir "%PROJECT%" 2>nul
mkdir "%PROJECT%\01-Baseline" 2>nul
mkdir "%PROJECT%\02-Batch-Normalization" 2>nul
mkdir "%PROJECT%\03-Comparison" 2>nul

echo.
echo Project structure created successfully.
echo.

tree "%PROJECT%" /F

echo.
pause