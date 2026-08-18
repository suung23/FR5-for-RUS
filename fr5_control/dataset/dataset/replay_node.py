import rclpy
from rclpy.node import Node
import h5py
import numpy as np
import os
import sys

# Message Imports
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32

class DataReplayNode(Node):
    def __init__(self):
        super().__init__('data_replay_node')

        # --- [설정] 파라미터 ---
        # 재생할 h5 파일 경로 (실행 시 파라미터로 지정 필요)
        self.declare_parameter('file_path', '')
        # 재생 빈도 (기본 30Hz, 녹화된 주기와 맞추는 것이 좋습니다)
        self.declare_parameter('replay_frequency', 30.0)

        file_path = self.get_parameter('file_path').value
        self.replay_freq = self.get_parameter('replay_frequency').value

        # 파일 경로 확인
        if not file_path or not os.path.exists(file_path):
            self.get_logger().error(f"File not found or not specified: {file_path}")
            self.get_logger().error("Usage: ros2 run <pkg> <node> --ros-args -p file_path:=/path/to/episode.h5")
            sys.exit(1)

        # --- HDF5 파일 로드 ---
        self.get_logger().info(f"Loading dataset: {file_path}")
        self.h5_file = h5py.File(file_path, 'r')
        
        # 그룹 키('0', '1', ...)를 정수 순서로 정렬
        # 문자열 정렬('1', '10', '2') 방지를 위해 int 변환 후 정렬
        self.sorted_keys = sorted(self.h5_file.keys(), key=lambda x: int(x))
        self.total_steps = len(self.sorted_keys)
        self.current_idx = 0

        self.get_logger().info(f"Total steps to replay: {self.total_steps}")

        # --- Publishers (요청하신 토픽 이름) ---
        
        # 1. Left Arm
        self.pub_left_pose = self.create_publisher(
            Pose, '/fr5_left/desired_pose_wrt_rcm', 10)
        self.pub_left_gripper = self.create_publisher(
            Float32, '/fr5_left/desired_gripper_pose', 10)

        # 2. Right Arm
        self.pub_right_pose = self.create_publisher(
            Pose, '/fr5_right/desired_pose_wrt_rcm', 10)
        self.pub_right_gripper = self.create_publisher(
            Float32, '/fr5_right/desired_gripper_pose', 10)

        # --- Timer ---
        timer_period = 1.0 / self.replay_freq
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info(f"Replay started at {self.replay_freq} Hz...")

    def timer_callback(self):
        if self.current_idx >= self.total_steps:
            self.get_logger().info("Replay finished successfully.")
            self.timer.cancel()
            self.h5_file.close()
            # 노드 종료 (선택 사항)
            raise SystemExit
            return

        # 현재 스텝의 데이터 가져오기
        step_key = self.sorted_keys[self.current_idx]
        step_data = self.h5_file[step_key]

        # --- 1. Left Arm Replay ---
        # Pose: fr5_left_gripper_wrt_rcm -> /fr5_left/desired_pose_wrt_rcm
        if 'fr5_left_gripper_wrt_rcm' in step_data:
            pose_arr = np.array(step_data['fr5_left_gripper_wrt_rcm'])
            msg = self.numpy_to_pose(pose_arr)
            self.pub_left_pose.publish(msg)

        # Gripper: fr5_left_desired_gripper_state -> /fr5_left/desired_gripper_pose
        if 'fr5_left_desired_gripper_state' in step_data:
            val = float(step_data['fr5_left_desired_gripper_state'][0])
            msg = Float32()
            msg.data = val
            self.pub_left_gripper.publish(msg)

        # --- 2. Right Arm Replay ---
        # Pose: fr5_right_gripper_wrt_rcm -> /fr5_right/desired_pose_wrt_rcm
        if 'fr5_right_gripper_wrt_rcm' in step_data:
            pose_arr = np.array(step_data['fr5_right_gripper_wrt_rcm'])
            msg = self.numpy_to_pose(pose_arr)
            self.pub_right_pose.publish(msg)

        # Gripper: fr5_right_desired_gripper_state -> /fr5_right/desired_gripper_pose
        if 'fr5_right_desired_gripper_state' in step_data:
            val = float(step_data['fr5_right_desired_gripper_state'][0])
            msg = Float32()
            msg.data = val
            self.pub_right_gripper.publish(msg)

        # 인덱스 증가
        self.current_idx += 1

    def numpy_to_pose(self, arr):
        """
        Numpy array (7,) [x, y, z, qx, qy, qz, qw] -> geometry_msgs/Pose
        """
        msg = Pose()
        msg.position.x = float(arr[0])
        msg.position.y = float(arr[1])
        msg.position.z = float(arr[2])
        msg.orientation.x = float(arr[3])
        msg.orientation.y = float(arr[4])
        msg.orientation.z = float(arr[5])
        msg.orientation.w = float(arr[6])
        return msg

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = DataReplayNode()
        rclpy.spin(node)
    except SystemExit:
        print("Replay Node Exited.")
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()