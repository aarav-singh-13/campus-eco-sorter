@echo off
setlocal
REM Uploads this project to GitHub. Create an EMPTY repo on github.com first (no README).
cd /d "%~dp0"

where git >nul 2>nul
if errorlevel 1 (
  echo Git is not installed. Install it from https://git-scm.com/download/win, then run this again.
  pause
  exit /b 1
)

git config user.name >nul 2>nul
if errorlevel 1 call :askname
git config user.email >nul 2>nul
if errorlevel 1 call :askemail

if not exist .git git init -b main

git add .
git commit -m "Campus Eco & Waste Sorting Identifier"

git remote get-url origin >nul 2>nul
if errorlevel 1 call :askremote

echo.
echo Uploading... (a browser window may open asking you to sign in to GitHub)
git push -u origin main
echo.
echo Done. Refresh your GitHub repo page to see the files.
pause
exit /b 0

:askname
set /p GN=Your name (for commits):
git config --global user.name "%GN%"
exit /b 0

:askemail
set /p GE=Your GitHub email:
git config --global user.email "%GE%"
exit /b 0

:askremote
set /p REPO=Paste your GitHub repo URL (like https://github.com/yourname/campus-eco-sorter.git):
git remote add origin %REPO%
exit /b 0
