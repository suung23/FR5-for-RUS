import h5py
import numpy as np
import math
import threading
import time
import sys
import os
import argparse

# Robot SDK Import
from fairino import Robot

# --- 설정 ---
ROBOT_IPS = {
    'fr5_left': '192.168.58.2',
    'fr5_right': '192.168.58.3'
}

def read_initial_joints_from_h5(h5_path):
    """
    H5 파일의 첫 번째 스텝(그룹 '0')에서 관절 각도를 읽어옵니다.
    Return: (left_joints_deg, right_joints_deg) 리스트
    """
    if not os.path.exists(h5_path):
        print(f"[Error] File not found: {h5_path}")
        sys.exit(1)

    try:
        with h5py.File(h5_path, 'r') as f:
            # 그룹 '0'이 첫 번째 스텝입니다.
            if '0' not in f:
                print("[Error] Group '0' (first step) not found in H5 file.")
                sys.exit(1)
            
            grp = f['0']
            
            # 데이터셋 키 확인 (collector 코드 기준)
            key_left = 'fr5_left_joint_states'
            key_right = 'fr5_right_joint_states'

            if key_left not in grp or key_right not in grp:
                print("[Error] Joint states data not found in group '0'.")
                sys.exit(1)

            # 데이터 읽기 (Radian 단위)
            left_rad = grp[key_left][:]
            right_rad = grp[key_right][:]

            # Radian -> Degree 변환
            left_deg = [math.degrees(x) for x in left_rad]
            right_deg = [math.degrees(x) for x in right_rad]

            print(f"[Data] Loaded from {h5_path}")
            print(f"       Left Target (Deg): {[round(x, 2) for x in left_deg]}")
            print(f"       Right Target (Deg): {[round(x, 2) for x in right_deg]}")

            return left_deg, right_deg

    except Exception as e:
        print(f"[Error] Failed to read H5 file: {e}")
        sys.exit(1)

def move_robot_task(robot_name, ip, target_joints_deg):
    """
    개별 로봇을 제어하여 목표 위치로 이동시키는 스레드 함수
    """
    print(f"[{robot_name}] Connecting to {ip}...")
    try:
        # 1. 연결
        robot = Robot.RPC(ip)
        
        # 2. 서보 모드 종료 (안전 장치)
        # Homing이나 PTP 이동 전에는 ServoMoveEnd를 호출하여 일반 모드로 전환해야 함
        robot.ServoMoveEnd()
        time.sleep(0.5)

        # 3. 이동 명령 (MoveJ)
        # vel=20.0 (속도 20%), blendT=-1.0 (동작 완료 후 정지)
        print(f"[{robot_name}] Moving to start position...")
        ret = robot.MoveJ(
            joint_pos=target_joints_deg,
            tool=0,
            user=0,
            vel=20.0,    # 속도 20% (안전을 위해 적당한 속도 설정)
            blendT=-1.0  # -1.0: Blocking (해당 지점에 정확히 멈춤)
        )

        if ret != 0:
            print(f"[{robot_name}] MoveJ Failed! Error Code: {ret}")
        else:
            # RPC 명령이 성공적으로 전송되었더라도 물리적 이동 완료를 위해 잠시 대기
            # (Blocking 모드라도 약간의 버퍼 시간 권장)
            time.sleep(0.5)
            print(f"[{robot_name}] Reached Start Position.")

    except Exception as e:
        print(f"[{robot_name}] Exception occurred: {e}")

def main():
    # 인자 파싱 (파일 경로 입력)
    parser = argparse.ArgumentParser(description="Reset FR5 robots to initial state from H5 file.")
    parser.add_argument('--file', type=str, help="Path to the .h5 file")
    args = parser.parse_args()

    h5_file_path = args.file

    # 1. H5 파일에서 초기 관절 각도 로드
    left_target, right_target = read_initial_joints_from_h5(h5_file_path)

    print("-" * 50)
    print("Starting Robot Reset Sequence...")
    print("-" * 50)

    # 2. 스레딩을 이용한 동시 이동
    t_left = threading.Thread(
        target=move_robot_task, 
        args=('fr5_left', ROBOT_IPS['fr5_left'], left_target)
    )
    t_right = threading.Thread(
        target=move_robot_task, 
        args=('fr5_right', ROBOT_IPS['fr5_right'], right_target)
    )

    # 시작
    t_left.start()
    t_right.start()

    # 완료 대기
    t_left.join()
    t_right.join()

    print("-" * 50)
    print("All robots initialized to H5 start position.")
    print("Exiting script.")

if __name__ == "__main__":
    main()