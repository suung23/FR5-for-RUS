@echo off
REM Send new collection sessions to the Linux training PC over the LAN (no git).
REM One-time key install (asks the Linux password once):
REM   ssh-keygen -t ed25519 -N "" -f %USERPROFILE%\.ssh\id_ed25519
REM   type %USERPROFILE%\.ssh\id_ed25519.pub | ssh rosotauser@192.168.77.1 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
cd /d %~dp0
python host\push_sessions.py %*
