@echo off
rem ============================================================================
rem  US(C10UR, Wi-Fi) + IMU(BNO085, COM 자동) 수집 시작 — Windows 노트북 (2026-09-09)
rem
rem    imu_bench\start_collect_win.cmd              기본: 1000 프레임 자동 정지, 부채꼴 표시, 자동 동기화
rem    imu_bench\start_collect_win.cmd 500          프레임 수 지정
rem
rem  순서:  1) 벤더 뷰어(WirelessUSG) 가 떠 있으면 종료 (프로브 TCP 슬롯을 하나만 받는다)
rem         2) 동글(Wi-Fi 2) 을 프로브 AP 에 접속하고 5002/5003 포트 확인 (안 열리면 동글 재연결로 프로브 세션 리셋)
rem         3) GUI 실행.  창을 클릭해 포커스 → Z(영점 3 s, 정지) → R(스캔+녹화) → 자동 정지 → R 로 다음 세션 …
rem            저장 완료 세션은 GUI 안의 루프가 5 s 안에 동기화 (sync.npz, sync_report.json, sync_check.png)
rem  세션 폴더: imu_bench\logs\us_imu_<시각>\
rem ============================================================================
setlocal
set HERE=%~dp0
set FRAMES=%1
if "%FRAMES%"=="" set FRAMES=1000
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo [1/3] 벤더 뷰어 종료 (있으면)
taskkill /IM WirelessUSG.exe /F >nul 2>&1

echo [2/3] 동글을 프로브 AP 에 접속하고 포트 확인
python "%HERE%host\probe_wifi_win.py" --connect-only
if errorlevel 4 (
  echo     포트가 닫혀 있어 동글을 재연결합니다 ...
  netsh wlan disconnect interface="Wi-Fi 2" >nul 2>&1
  timeout /t 3 /nobreak >nul
  python "%HERE%host\probe_wifi_win.py" --connect-only
  if errorlevel 4 (
    echo     여전히 닫혀 있습니다. 프로브 전원을 껐다 켠 뒤 다시 실행하십시오.
    exit /b 4
  )
)
if errorlevel 1 (
  echo     프로브 AP 를 찾지 못했거나 접속 실패 — 프로브 전원^(배터리^), 동글 연결 확인
  exit /b 2
)

echo [3/3] GUI 실행  (프레임 %FRAMES% 자동 정지)
python "%HERE%host\us_imu_gui.py" --probe c10ur --host 192.168.1.1 --record-frames %FRAMES% --out-dir "%HERE%logs"
endlocal
