import rclpy
from rclpy.node import Node
import h5py
import numpy as np
import cv2
from cv_bridge import CvBridge
import time
import os
from datetime import datetime
import signal
import sys

# Message Imports
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import Pose, Twist

class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('data_collector_node')
        
        # --- [설정] 파라미터 ---
        self.declare_parameter('dataset_dir', 'collected_data')
        self.declare_parameter('collect_frequency', 30.0)

        self.output_dir = self.get_parameter('dataset_dir').value
        self.collect_freq = self.get_parameter('collect_frequency').value
        
        # 폴더 생성
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.get_logger().info(f"Dataset will be saved to: {self.output_dir}")
        except Exception as e:
            self.get_logger().error(f"Failed to create dir: {e}")
            self.output_dir = "."
        
        self.jpeg_quality = 90 
        
        # --- State Variables ---
        self.bridge = CvBridge()
        self.recording_started = False
        self.step_count = 0  # 현재 저장된 스텝 수
        
        # [수정 1] 파일 이름 생성 및 미리 열기
        filename = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        self.filepath = os.path.join(self.output_dir, filename)
        
        try:
            self.h5_file = h5py.File(self.filepath, 'w')
            self.get_logger().info(f"Created H5 file: {self.filepath}")
        except Exception as e:
            self.get_logger().error(f"Failed to create file: {e}")
            sys.exit(1)

        # 데이터 버퍼 (None으로 초기화)
        self.latest_data = {
            "image1": None,
            "fr5_left_joint_states": None, "fr5_left_gripper_state": None,
            "fr5_left_gripper_wrt_rcm": None, "fr5_left_gripper_wrt_right_gripper": None,
            "fr5_left_ee_wrt_base": None,
            "fr5_right_joint_states": None, "fr5_right_gripper_state": None,
            "fr5_right_gripper_wrt_rcm": None, "fr5_right_gripper_wrt_left_gripper": None,
            "fr5_right_ee_wrt_base": None,
            "fr5_left_desired_gripper_state": None, "fr5_right_desired_gripper_state": None
        }

        # --- Subscribers (기존과 동일) ---
        self.create_subscription(Image, '/camera/image1', self.image_callback, 10)

        self.create_subscription(JointState, '/fr5_left/joint_states', lambda m: self.update_joint('fr5_left_joint_states', m), 10)
        self.create_subscription(Float32, '/fr5_left/gripper_state', lambda m: self.update_scalar('fr5_left_gripper_state', m), 10)
        self.create_subscription(Pose, '/fr5_left/gripper_wrt_rcm', lambda m: self.update_pose('fr5_left_gripper_wrt_rcm', m), 10)
        self.create_subscription(Pose, '/fr5/left_gripper_wrt_right_gripper', lambda m: self.update_pose('fr5_left_gripper_wrt_right_gripper', m), 10)
        self.create_subscription(Float32, '/fr5_left/desired_gripper_pose', lambda m: self.update_scalar('fr5_left_desired_gripper_state', m), 10)
        self.create_subscription(Pose, '/fr5_left/ee_wrt_base', lambda m: self.update_pose('fr5_left_ee_wrt_base', m), 10)

        self.create_subscription(JointState, '/fr5_right/joint_states', lambda m: self.update_joint('fr5_right_joint_states', m), 10)
        self.create_subscription(Float32, '/fr5_right/gripper_state', lambda m: self.update_scalar('fr5_right_gripper_state', m), 10)
        self.create_subscription(Pose, '/fr5_right/gripper_wrt_rcm', lambda m: self.update_pose('fr5_right_gripper_wrt_rcm', m), 10)
        self.create_subscription(Pose, '/fr5/right_gripper_wrt_left_gripper', lambda m: self.update_pose('fr5_right_gripper_wrt_left_gripper', m), 10)
        self.create_subscription(Float32, '/fr5_right/desired_gripper_pose', lambda m: self.update_scalar('fr5_right_desired_gripper_state', m), 10)
        self.create_subscription(Pose, '/fr5_right/ee_wrt_base', lambda m: self.update_pose('fr5_right_ee_wrt_base', m), 10)

        # Timer
        timer_period = 1.0 / self.collect_freq
        self.timer = self.create_timer(timer_period, self.collect_timer_cb)
        
        self.get_logger().info("Data Collector Ready.")

    # --- Update Callbacks (기존과 동일) ---
    def update_joint(self, key, msg): self.latest_data[key] = np.array(msg.position, dtype=np.float32)
    def update_scalar(self, key, msg): self.latest_data[key] = np.array([msg.data], dtype=np.float32)
    def update_pose(self, key, msg):
        arr = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w], dtype=np.float32)
        self.latest_data[key] = arr
    def image_callback(self, msg):
        try: self.latest_data['image1'] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e: self.get_logger().error(f"RGB fail: {e}")


    # --- Timer Callback: Save Immediately ---
    def collect_timer_cb(self):
        if any(v is None for v in self.latest_data.values()):
            return

        if not self.recording_started:
            self.get_logger().info("Starting recording...")
            self.recording_started = True

        # [수정 2] 데이터를 모으지 않고 즉시 저장 함수 호출
        self.save_step(self.latest_data)

    def save_step(self, raw_data):
        """ 한 스텝의 데이터를 H5 파일에 즉시 기록 """
        try:
            # 그룹 이름: 0, 1, 2 ...
            grp_name = str(self.step_count)
            grp = self.h5_file.create_group(grp_name)

            # 1. Timestamp
            grp.create_dataset('timestamp', data=np.array([time.time()], dtype=np.float64))

            # 2. Image (Compress & Save)
            success, encoded_img = cv2.imencode('.jpg', raw_data['image1'], [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if success:
                grp.create_dataset('image1', data=encoded_img)
            else:
                grp.create_dataset('image1', data=np.array([], dtype=np.uint8))

            # 3. Vectors
            keys_to_copy = [
                'fr5_left_joint_states', 'fr5_left_gripper_state', 
                'fr5_left_gripper_wrt_rcm', 'fr5_left_gripper_wrt_right_gripper',
                'fr5_right_joint_states', 'fr5_right_gripper_state', 
                'fr5_right_gripper_wrt_rcm', 'fr5_right_gripper_wrt_left_gripper',
                'fr5_left_desired_gripper_state', 'fr5_right_desired_gripper_state',
                'fr5_left_ee_wrt_base', 'fr5_right_ee_wrt_base',
            ]
            
            for k in keys_to_copy:
                grp.create_dataset(k, data=raw_data[k])

            # [중요] 매번 flush()를 호출하여 디스크에 확실히 기록
            self.h5_file.flush()
            
            self.step_count += 1
            if self.step_count % 30 == 0:
                print(f"Recorded step {self.step_count}...", end='\r')

        except Exception as e:
            self.get_logger().error(f"Error saving step: {e}")

    def close_file(self):
        if hasattr(self, 'h5_file') and self.h5_file:
            self.get_logger().info(f"Closing file. Total steps: {self.step_count}")
            self.h5_file.close()

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # [수정 3] 종료 시 파일만 안전하게 닫음
        node.close_file()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()