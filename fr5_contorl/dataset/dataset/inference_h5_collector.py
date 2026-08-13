import rclpy
from rclpy.node import Node
import h5py
import numpy as np
import cv2
from cv_bridge import CvBridge
import time
import os
from datetime import datetime
import sys

# Message Imports
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32, Float32MultiArray, String
from geometry_msgs.msg import Pose

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
        self.step_count = 0  
        
        # 파일 이름 생성 및 HDF5 파일 열기
        base_filename = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.filepath = os.path.join(self.output_dir, f"{base_filename}.h5")
        
        # 비디오 저장 경로 및 Writer 변수 초기화
        self.video_filepath = os.path.join(self.output_dir, f"{base_filename}.mp4")
        self.video_writer = None
        
        try:
            self.h5_file = h5py.File(self.filepath, 'w')
            self.get_logger().info(f"Created H5 file: {self.filepath}")
        except Exception as e:
            self.get_logger().error(f"Failed to create file: {e}")
            sys.exit(1)

        # 데이터 버퍼 (None으로 초기화)
        self.latest_data = {
            "image1": None, 
            "depth_map": None,
            "fr5_left_joint_states": None, "fr5_left_gripper_state": None,
            "fr5_left_gripper_wrt_rcm": None, "fr5_left_gripper_wrt_right_gripper": None,
            "fr5_left_ee_wrt_base": None,
            "fr5_right_joint_states": None, "fr5_right_gripper_state": None,
            "fr5_right_gripper_wrt_rcm": None, "fr5_right_gripper_wrt_left_gripper": None,
            "fr5_right_ee_wrt_base": None,
            "fr5_left_desired_gripper_state": None, "fr5_right_desired_gripper_state": None,
            "needle_location_vec": None,
            "predicted_task": None
        }

        # --- Subscribers ---
        self.create_subscription(Image, '/camera/image1', self.image_callback, 10)
        self.create_subscription(Image, '/depth_map', self.depth_map_callback, 10) 

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

        self.create_subscription(Float32MultiArray, '/clicked_needle_points', self.needle_points_callback, 10)
        self.create_subscription(String, 'predicted_task', self.predicted_task_callback, 10)

        # --- Timers ---
        # 1. 실제 데이터를 수집하고 저장하는 메인 타이머 (예: 30Hz)
        timer_period = 1.0 / self.collect_freq
        self.timer = self.create_timer(timer_period, self.collect_timer_cb)
        
        # [추가됨] 2. 1초마다 누락된 데이터를 알려주는 디버그 타이머 (1.0Hz)
        self.debug_timer = self.create_timer(1.0, self.debug_missing_data_cb)
        
        self.get_logger().info("Data Collector Ready. Waiting for all topics to arrive...")

    # --- Update Callbacks ---
    def update_joint(self, key, msg): self.latest_data[key] = np.array(msg.position, dtype=np.float32)
    def update_scalar(self, key, msg): self.latest_data[key] = np.array([msg.data], dtype=np.float32)
    def update_pose(self, key, msg):
        arr = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w], dtype=np.float32)
        self.latest_data[key] = arr
    
    def image_callback(self, msg):
        try: self.latest_data['image1'] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e: self.get_logger().error(f"RGB fail: {e}")
        
    def depth_map_callback(self, msg):
        try: self.latest_data['depth_map'] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e: self.get_logger().error(f"Depth Map fail: {e}")
            

    
    def needle_points_callback(self, msg):
        self.latest_data['needle_location_vec'] = np.array(msg.data, dtype=np.float32)
    def predicted_task_callback(self, msg):
        self.latest_data['predicted_task'] = msg.data

    # --- [추가됨] Debug Timer Callback ---
    def debug_missing_data_cb(self):
        # 만약 아직 수집이 시작되지 않았다면, 어떤 데이터가 None인지 확인해서 로깅합니다.
        if not self.recording_started:
            missing_keys = [key for key, value in self.latest_data.items() if value is None]
            if missing_keys:
                self.get_logger().warn(f"Waiting for topics... Missing data: {missing_keys}")

    # --- Timer Callback: Save Immediately ---
    def collect_timer_cb(self):
        # 모든 데이터가 None이 아니어야 저장을 시작함
        if any(v is None for v in self.latest_data.values()):
            return

        if not self.recording_started:
            self.get_logger().info("✅ All topics received! Starting recording...")
            self.recording_started = True

        self.save_step(self.latest_data)

    def save_step(self, raw_data):
        """ 한 스텝의 데이터를 H5 파일과 MP4 비디오에 기록 """
        try:
            # VideoWriter 초기화 (첫 프레임 기준)
            if self.video_writer is None:
                height, width, _ = raw_data['image1'].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                self.video_writer = cv2.VideoWriter(self.video_filepath, fourcc, float(self.collect_freq), (width, height))
                self.get_logger().info(f"Created MP4 file: {self.video_filepath}")

            # 원본 RGB 영상 프레임 기록
            self.video_writer.write(raw_data['image1'])

            # 그룹 이름: 0, 1, 2 ...
            grp_name = str(self.step_count)
            grp = self.h5_file.create_group(grp_name)

            # 1. Timestamp
            grp.create_dataset('timestamp', data=np.array([time.time()], dtype=np.float64))

            # 2. RGB Image (JPEG 압축)
            success_rgb, encoded_img = cv2.imencode('.jpg', raw_data['image1'], [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if success_rgb:
                grp.create_dataset('image1', data=encoded_img)
            else:
                grp.create_dataset('image1', data=np.array([], dtype=np.uint8))

            # 3. Depth Map (PNG 무손실 압축)
            success_depth, encoded_depth = cv2.imencode('.png', raw_data['depth_map'])
            if success_depth:
                grp.create_dataset('depth_map', data=encoded_depth)
            else:
                grp.create_dataset('depth_map', data=np.array([], dtype=np.uint8))

            # 4. Vector & String 데이터
            keys_to_copy = [
                'fr5_left_joint_states', 'fr5_left_gripper_state', 
                'fr5_left_gripper_wrt_rcm', 'fr5_left_gripper_wrt_right_gripper',
                'fr5_right_joint_states', 'fr5_right_gripper_state', 
                'fr5_right_gripper_wrt_rcm', 'fr5_right_gripper_wrt_left_gripper',
                'fr5_left_desired_gripper_state', 'fr5_right_desired_gripper_state',
                'fr5_left_ee_wrt_base', 'fr5_right_ee_wrt_base',
                'needle_location_vec', 'predicted_task'
            ]
            
            for k in keys_to_copy:
                grp.create_dataset(k, data=raw_data[k])

            # 디스크 기록
            self.h5_file.flush()
            
            self.step_count += 1
            if self.step_count % 30 == 0:
                print(f"Recorded step {self.step_count}...", end='\r')

        except Exception as e:
            self.get_logger().error(f"Error saving step: {e}")

    def close_file(self):
        # H5 파일 닫기
        if hasattr(self, 'h5_file') and self.h5_file:
            self.get_logger().info(f"Closing H5 file. Total steps: {self.step_count}")
            self.h5_file.close()
            
        # 비디오 파일 닫기
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info(f"Saved Video file: {self.video_filepath}")

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_file()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()