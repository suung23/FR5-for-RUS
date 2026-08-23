import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import threading
from matplotlib.lines import Line2D 

# --- ROS2 추가 임포트 ---
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32  # Float32 메시지 타입 임포트

# ==========================================
# 0. 전역 변수 및 스레드 동기화 설정
# ==========================================
q_ref = None
TOTAL_INSTRUMENT_LENGTH = 310.0

# 최신 데이터를 저장할 공유 딕셔너리와 Lock 객체
data_lock = threading.Lock()
latest_data = {
    'current_quat': [0, 0, 0, 1],
    'dist': 0.0,
    'grip': 0.0,
    'has_new': False,
    'is_running': True
}

# ==========================================
# 1. 시리얼 포트 연결 로직
# ==========================================
def select_serial_port():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("❌ 연결된 시리얼 포트가 없습니다.")
        sys.exit()

    print("\n=== 사용 가능한 시리얼 포트 목록 ===")
    for i, port in enumerate(ports):
        print(f"[{i + 1}] {port.device} - {port.description}")
    print("===================================\n")

    while True:
        try:
            choice = input("연결할 장치의 번호를 선택하세요 (q: 종료): ")
            if choice.lower() == 'q': sys.exit()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(ports):
                return ports[choice_idx].device
            else:
                print("⚠️ 유효한 번호를 입력해 주세요.")
        except ValueError:
            print("⚠️ 숫자를 입력해 주세요.")

PORT = select_serial_port()
BAUD_RATE = 115200

try:
    ser = serial.Serial(PORT, BAUD_RATE, timeout=0.1)
    print(f"\n✅ {PORT} 포트에 연결되었습니다.")
except Exception as e:
    print(f"❌ 연결 실패: {e}")
    sys.exit()

# ==========================================
# [추가] ROS2 노드 및 퍼블리셔 초기화
# ==========================================
rclpy.init(args=None)
ros_node = rclpy.create_node('laparoscopic_sensor_publisher')

# 기존 Pose 퍼블리셔
pose_pub = ros_node.create_publisher(Pose, '/fr5_right/desired_pose_wrt_rcm', 10)
# 신규 그리퍼 값 퍼블리셔 (Float32)
grip_pub = ros_node.create_publisher(Float32, '/fr5_right/desired_gripper_pose', 10)

print("🌐 ROS2 Node Started:")
print("   - Publishing Pose to '/fr5_right/desired_pose_wrt_rcm'")
print("   - Publishing Gripper to '/fr5_right/desired_gripper_pose'")

# ==========================================
# 2. 백그라운드 시리얼 수신 스레드
# ==========================================
def serial_read_thread():
    global latest_data
    while latest_data['is_running']:
        try:
            # 버퍼에 데이터가 있으면 계속 읽어서 가장 최신 상태로 업데이트
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                data = line.split(',')
                if len(data) == 7:
                    t, w, x, y, z, dist, grip = map(float, data)
                    
                    # 공유 데이터 업데이트 시 충돌을 막기 위해 Lock 사용
                    with data_lock:
                        latest_data['current_quat'] = [x, y, z, w]
                        latest_data['dist'] = dist
                        latest_data['grip'] = grip
                        latest_data['has_new'] = True
        except Exception:
            pass # 파싱 에러나 기타 에러는 무시하고 다음 줄 읽기 진행

# 데몬 스레드로 실행하여 메인 프로그램 종료 시 함께 종료되도록 설정
thread = threading.Thread(target=serial_read_thread, daemon=True)
thread.start()

# ==========================================
# 3. GUI 초기화
# ==========================================
fig = plt.figure(figsize=(14, 7))
fig.canvas.manager.set_window_title('Laparoscopic Instrument Dashboard')

ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
ax_dist = fig.add_subplot(2, 2, 2)
ax_grip = fig.add_subplot(2, 2, 4)

def on_key(event):
    global q_ref
    if event.key == 'c' or event.key == 'C':
        with data_lock:
            q_ref = list(latest_data['current_quat'])
        print("\n🎯 [캘리브레이션 완료] 현재 방향을 0점으로 설정했습니다!\n")

fig.canvas.mpl_connect('key_press_event', on_key)

# ==========================================
# 4. GUI 업데이트 함수 (30Hz로 호출됨)
# ==========================================
def update(frame):
    global q_ref
    
    # 1. Lock을 걸고 최신 데이터를 안전하게 복사
    with data_lock:
        if not latest_data['has_new']:
            return # 새로운 데이터가 없으면 이번 프레임은 스킵
        
        current_quat = list(latest_data['current_quat'])
        dist = latest_data['dist']
        grip = latest_data['grip']
        latest_data['has_new'] = False # 처리했으므로 플래그 초기화

    # ROS2 이벤트 루프 처리 (퍼블리셔 유지보수용)
    rclpy.spin_once(ros_node, timeout_sec=0)

    # 2. 캘리브레이션(영점) 처리
    if q_ref is None:
        q_ref = current_quat
        print("✅ 첫 데이터 수신 완료: 시작 위치를 기준 좌표계로 설정했습니다.")

    # 3. 3D 위치 및 방향 렌더링
    ax_3d.cla() 
    ax_3d.view_init(elev=20, azim=200) 
    limit = 400 
    ax_3d.set_xlim([-limit, limit])
    ax_3d.set_ylim([-limit, limit])
    ax_3d.set_zlim([-limit, limit])
    ax_3d.set_title("Instrument Kinematics (Press 'c' to zero)")

    ax_3d.set_xlabel('X-axis (mm)')
    ax_3d.set_ylabel('Y-axis (mm)')
    ax_3d.set_zlabel('Z-axis (mm)')
    
    ax_3d.plot([-limit, limit], [0, 0], [0, 0], 'black', linestyle='--', alpha=0.2)
    ax_3d.plot([0, 0], [-limit, limit], [0, 0], 'black', linestyle='--', alpha=0.2)
    ax_3d.plot([0, 0], [0, 0], [-limit, limit], 'black', linestyle='--', alpha=0.2)
    
    r_current = R.from_quat(current_quat)
    r_ref = R.from_quat(q_ref)
    r_rel = r_ref.inv() * r_current
    
    base_axes = np.eye(3) 
    rotated_axes = r_rel.apply(base_axes)
    inst_x_axis = rotated_axes[0] 
    
    gripper_dist = TOTAL_INSTRUMENT_LENGTH - dist
    gripper_pos = gripper_dist * inst_x_axis  
    imu_pos = -dist * inst_x_axis             
    
    # ==========================================
    # ROS2 퍼블리시 파트
    # ==========================================
    # 1. 위치 및 방향 (Pose) 퍼블리시
    pose_msg = Pose()
    pose_msg.position.x = float(gripper_pos[0] / 1000.0)
    pose_msg.position.y = float(gripper_pos[1] / 1000.0)
    pose_msg.position.z = float(gripper_pos[2] / 1000.0)
    
    quat = r_rel.as_quat()
    pose_msg.orientation.x = float(quat[0])
    pose_msg.orientation.y = float(quat[1])
    pose_msg.orientation.z = float(quat[2])
    pose_msg.orientation.w = float(quat[3])
    
    pose_pub.publish(pose_msg)

    # 2. [추가] 그리퍼 상태 (Float32) 퍼블리시
    # 50%를 초과하면 100.0, 이하면 5.0 전송
    grip_msg = Float32()
    if grip > 50.0:
        grip_msg.data = 5.0
    else:
        grip_msg.data = 100.0
        
    grip_pub.publish(grip_msg)
    # ==========================================

    # 시각화 오브젝트 그리기
    ax_3d.scatter(0, 0, 0, color='black', s=80, marker='o', label='Trocar (Origin)')
    ax_3d.plot([imu_pos[0], gripper_pos[0]], 
               [imu_pos[1], gripper_pos[1]], 
               [imu_pos[2], gripper_pos[2]], 
               color='gray', linewidth=4, alpha=0.7)
    
    colors = ['r', 'g', 'b']
    labels = ['+X', '+Y', '+Z']
    arrow_length = 80 
    
    for i in range(3):
        ax_3d.quiver(*gripper_pos, *rotated_axes[i], length=arrow_length, color=colors[i], normalize=True, linewidth=3)
        text_pos = gripper_pos + (rotated_axes[i] * arrow_length * 1.2)
        ax_3d.text(*text_pos, labels[i], color=colors[i], fontsize=10, fontweight='bold')

    handles, labels_plot = ax_3d.get_legend_handles_labels()
    legend_elements = [
        Line2D([0], [0], color='gray', lw=4, alpha=0.7, label='Shaft'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Trocar (0,0,0)'),
        Line2D([0], [0], color='r', lw=3, label='Gripper X+'),
        Line2D([0], [0], color='g', lw=3, label='Gripper Y+'),
        Line2D([0], [0], color='b', lw=3, label='Gripper Z+')
    ]
    ax_3d.legend(handles=legend_elements, loc='upper left', fontsize=8)

    # 4. 센서 바 그래프 렌더링
    ax_dist.cla()
    ax_dist.barh(['Insertion Depth (mm)'], [gripper_dist], color='cornflowerblue')
    ax_dist.set_xlim([-50, 350]) 
    ax_dist.set_title(f"Gripper Pos from Trocar: {gripper_dist:.1f} mm")
    ax_dist.text(max(0, gripper_dist) + 5, 0, f"{gripper_dist:.1f}", va='center', fontweight='bold')

    ax_grip.cla()
    ax_grip.barh(['Gripper (%)'], [grip], color='coral')
    ax_grip.set_xlim([0, 100]) 
    
    # 터미널이나 로봇에 실제로 전달되는 값을 시각적으로도 보여주면 디버깅에 좋습니다.
    actual_pub_val = 100.0 if grip > 50.0 else 5.0
    ax_grip.set_title(f"Gripper Opened: {int(grip)} % (Pub: {actual_pub_val})")
    ax_grip.text(grip + 2, 0, f"{int(grip)}", va='center', fontweight='bold')

# ==========================================
# 5. 프로그램 실행 (30Hz = 약 33ms 주기)
# ==========================================
# interval=33 설정으로 30fps 유지
ani = animation.FuncAnimation(fig, update, interval=33, cache_frame_data=False)

plt.tight_layout()
try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    # 창을 닫을 때 스레드와 시리얼 포트, ROS2 안전하게 종료
    latest_data['is_running'] = False
    ser.close()
    
    ros_node.destroy_node()
    rclpy.shutdown()
    print("프로그램을 종료합니다.")