@echo off
rem ============================================================================
rem  US (C10UR over Wi-Fi) + IMU (BNO085, COM auto) collection - Windows laptop
rem  ASCII only: cmd.exe parses this file in the console code page (CP949),
rem  so Korean text here would be mangled and executed as commands.
rem
rem    imu_bench\start_collect_win.cmd                         stop after 1000 frames (default)
rem    imu_bench\start_collect_win.cmd 500                     custom frame count
rem    imu_bench\start_collect_win.cmd manual                  no auto stop: R starts, R stops (any length)
rem    imu_bench\start_collect_win.cmd manual find_bladder 100 episode mode: one session = one episode,
rem                                                            task name + target count shown in the GUI,
rem                                                            M = "target found" mark, X = discard episode
rem
rem  Steps: 0) check the IMU firmware tag (host\flash_win.py --check). Old firmware = gyro calibration
rem            off = cal_gyr 0 / cal_rv 1 forever. Flash with: flash_win.py --uf2 <file> (UF2 built on
rem            Linux by scripts/build_uf2.sh; this laptop cannot run unsigned compilers - Smart App Control)
rem         1) kill the vendor viewer if running (probe accepts ONE tcp client)
rem         2) connect the dongle (Wi-Fi 2) to the probe AP, ping only
rem            (never open the probe ports here - the GUI opens them)
rem         3) run the GUI: click window -> K (IMU calibration guide: gyro rest, accel 6 poses,
rem            magnetometer figure-8 warm-up until rv = 3) -> S (save DCD to the BNO085 flash, survives
rem            power cycles) -> Z (zero, hold still 3 s; adopted even if the still check fails) -> R (scan +
rem            record, starts right away - missing firmware/rv 3/zero are only warnings, logged in the meta).
rem            -> auto stop at N frames, or R again in manual mode -> R again for the next session.
rem            Episode protocol (find_bladder): start OFF the bladder, hold 1 s, R, search with
rem            stop-move-stop (stops >= 0.5 s), press M when the bladder is in view, hold 1-2 s, R.
rem            Saved sessions are synchronized in the background (sync.npz, sync_report.json)
rem  Sessions: imu_bench\logs\us_imu_<stamp>\   (discarded: imu_bench\logs\discard_us_imu_<stamp>\)
rem ============================================================================
setlocal
set HERE=%~dp0
set FRAMES=%1
if "%FRAMES%"=="" set FRAMES=1000
if /I "%FRAMES%"=="manual" set FRAMES=0
if /I "%FRAMES%"=="0" set FRAMES=0
set TASKARGS=
if not "%2"=="" set TASKARGS=--task %2
if not "%3"=="" set TASKARGS=%TASKARGS% --episodes %3
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo [0/3] IMU firmware check
python "%HERE%host\flash_win.py" --check
if not errorlevel 1 goto fw_ok
rem  No parentheses inside this block: cmd.exe would take a ")" in an echo as the end of an "if (" block.
echo.
echo     ***** OLD IMU FIRMWARE - gyro calibration is OFF, every session would log cal_gyr 0 / cal_rv 1 *****
echo     ***** flash first:  python imu_bench\host\flash_win.py --uf2 imu_bench\firmware\umi_device_hardware\umi_device_hardware.uf2 *****
echo     ***** R still works - the warning is logged in the session meta                                 *****
echo.
:fw_ok

echo [1/3] stopping vendor viewer if running
taskkill /IM WirelessUSG.exe /F >nul 2>&1

echo [2/3] connecting dongle to probe AP (ping only)
python "%HERE%host\probe_wifi_win.py" --connect-only
set RC=%ERRORLEVEL%
if "%RC%"=="4" (
  echo     no ping - re-associating the dongle ...
  netsh wlan disconnect interface="Wi-Fi 2" >nul 2>&1
  timeout /t 3 /nobreak >nul
  python "%HERE%host\probe_wifi_win.py" --connect-only
  set RC=%ERRORLEVEL%
)
if not "%RC%"=="0" (
  echo     probe not reachable ^(rc=%RC%^). Power-cycle the probe, check the dongle, then retry.
  exit /b %RC%
)

if "%FRAMES%"=="0" (
  echo [3/3] starting GUI ^(manual stop: R starts, R stops^) %TASKARGS%. Keys: K calib, Z zero, R record, M found, X discard, F scan, Q quit
) else (
  echo [3/3] starting GUI ^(auto stop at %FRAMES% frames^). Keys: K calib guide, Z zero, R record, F scan, Q quit
)
python "%HERE%host\us_imu_gui.py" --probe c10ur --host 192.168.1.1 --record-frames %FRAMES% %TASKARGS% --out-dir "%HERE%logs"
endlocal
