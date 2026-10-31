
@echo off
cd /d F:\Projects\machin-learning2

git init

git status
git add .

git diff --cached --quiet
if %errorlevel%==0 (
    echo No changes to commit.
) else (
    git commit -m "Update project"
)

git branch -M main

git remote get-url origin >nul 2>&1
if errorlevel 1 (
    git remote add origin https://github.com/hosseinhaddad/machin-learning2.git
)

git push -u origin main

pause