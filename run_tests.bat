@echo off
setlocal

cd /d "%~dp0"

echo Installing test dependencies (pytest, numpy, kivy, kivymd)...
python -m pip install -q pytest numpy kivy kivymd
if errorlevel 1 (
    echo Failed to install dependencies.
    exit /b 1
)

echo.
echo Running OKF test suite...
python -m pytest tests -v
set TEST_RESULT=%errorlevel%

echo.
if %TEST_RESULT% equ 0 (
    echo All tests passed.
) else (
    echo Tests FAILED.
)

exit /b %TEST_RESULT%
