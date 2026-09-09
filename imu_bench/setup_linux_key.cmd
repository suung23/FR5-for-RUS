@echo off
REM One-time: install this laptop's SSH key on the Linux training PC, then send all sessions over the LAN.
REM Asks the Linux password ONCE. Afterwards push_sessions_win.cmd needs no password.
set HOST=%1
if "%HOST%"=="" set HOST=192.168.77.1
cd /d %~dp0
if not exist "%USERPROFILE%\.ssh\id_ed25519" ssh-keygen -t ed25519 -N "" -f "%USERPROFILE%\.ssh\id_ed25519"
echo Installing key on rosotauser@%HOST% (enter the Linux password once)...
type "%USERPROFILE%\.ssh\id_ed25519.pub" | ssh -o StrictHostKeyChecking=accept-new rosotauser@%HOST% "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
if errorlevel 1 (echo Key install failed. Check the cable, the address and the password. & exit /b 1)
echo Key installed. Sending sessions...
python host\push_sessions.py --host %HOST% %2 %3 %4
