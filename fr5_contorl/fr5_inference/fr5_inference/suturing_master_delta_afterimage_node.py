#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import numpy as np
import cv2
from cv_bridge import CvBridge
import time
import sys
import threading
import requests
import json
import base64
from collections import deque
from scipy.spatial.transform import Rotation as R

# Message Imports
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32, String, Float32MultiArray
from geometry_msgs.msg import Pose, Twist 

# Service Imports
from suturing_interfaces.srv import RelativeDepthMapService

class SuturingMasterNode(Node):
    def __init__(self):
        super().__init__('suturing_master_node')
        
        # ==========================================
        # 1. 파라미터 및 설정
        # ==========================================
        self.declare_parameter('control_frequency', 20.0) 
        self.declare_parameter('n_obs_steps',4)          
        self.declare_parameter('inference_url', 'http://localhost:8001/inference')

        self.control_freq = self.get_parameter('control_frequency').value
        self.n_obs_steps = self.get_parameter('n_obs_steps').value
        self.inference_url = self.get_parameter('inference_url').value
        
        # 이미지 설정
        self.IMG_W = 640.0
        self.IMG_H = 480.0

        # 리사이즈 목표 크기
        self.TARGET_W = 320
        self.TARGET_H = 240

        self.DISPLAY_SCALE = 2.0  
        
        self.bridge = CvBridge()
        self.callback_group = ReentrantCallbackGroup() 

        self.task_map = {
            "Needle_Pickup": 0, "Needle_Handoff": 1, "Needle_Insertion": 2,
            "Needle_Extraction": 3, "Knot_Tying": 4
        }

        # ==========================================
        # [NEW] 그리퍼 상태 관리 변수 (3초 락 기능용)
        # ==========================================
        self.GRIP_LOCK_DURATION = 3.0  # 상태 변경 후 유지 시간 (초)
        
        self.ARM_PAUSE_DURATION = 2.0  # 그리퍼 동작 후 2초간 팔 정지
        self.arm_stop_until = 0.0      # 정지가 끝나는 시간 타임스탬프

        # Left Arm Gripper State
        self.last_left_grip_val = 100.0  # 초기값 Open
        self.last_left_grip_time = 0.0   # 마지막 변경 시각

        # Right Arm Gripper State
        self.last_right_grip_val = 100.0 # 초기값 Open
        self.last_right_grip_time = 0.0  # 마지막 변경 시각

        # ==========================================
        # 2. 데이터 버퍼
        # ==========================================
        self.latest_data = {
            "task_string": "Needle_Pickup",          
            "image1": None,               
            "fr5_left_joint_states": None, "fr5_left_gripper_state": None,
            "fr5_left_gripper_wrt_rcm": None, "fr5_left_gripper_wrt_right_gripper": None,
            "fr5_right_joint_states": None, "fr5_right_gripper_state": None,
            "fr5_right_gripper_wrt_rcm": None, "fr5_right_gripper_wrt_left_gripper": None,
            "needle_location_vec": np.zeros(4, dtype=np.float32) 
        }

        self.obs_buffer = deque(maxlen=self.n_obs_steps)

        # UI 및 제어 플래그
        self.clicked_points = [] 
        self.is_inference_running = False 
        self.current_afterimage_display = None # [NEW] GUI 표시용 원본 크기 잔상 이미지 변수

        # ==========================================
        # 3. Subscribers
        # ==========================================
        self.create_subscription(String, '/predicted_task', self.task_callback, 10, callback_group=self.callback_group)
        self.create_subscription(Image, '/camera/image1', self.image_callback, 10, callback_group=self.callback_group)
        
        # Left Arm State
        self.create_subscription(JointState, '/fr5_left/joint_states', lambda m: self.update_joint('fr5_left_joint_states', m), 10, callback_group=self.callback_group)
        self.create_subscription(Float32, '/fr5_left/gripper_state', lambda m: self.update_scalar('fr5_left_gripper_state', m), 10, callback_group=self.callback_group)
        self.create_subscription(Pose, '/fr5_left/gripper_wrt_rcm', lambda m: self.update_pose('fr5_left_gripper_wrt_rcm', m), 10, callback_group=self.callback_group)
        self.create_subscription(Pose, '/fr5/left_gripper_wrt_right_gripper', lambda m: self.update_pose('fr5_left_gripper_wrt_right_gripper', m), 10, callback_group=self.callback_group)
        
        # Right Arm State
        self.create_subscription(JointState, '/fr5_right/joint_states', lambda m: self.update_joint('fr5_right_joint_states', m), 10, callback_group=self.callback_group)
        self.create_subscription(Float32, '/fr5_right/gripper_state', lambda m: self.update_scalar('fr5_right_gripper_state', m), 10, callback_group=self.callback_group)
        self.create_subscription(Pose, '/fr5_right/gripper_wrt_rcm', lambda m: self.update_pose('fr5_right_gripper_wrt_rcm', m), 10, callback_group=self.callback_group)
        self.create_subscription(Pose, '/fr5/right_gripper_wrt_left_gripper', lambda m: self.update_pose('fr5_right_gripper_wrt_left_gripper', m), 10, callback_group=self.callback_group)

        # ==========================================
        # 4. Publishers
        # ==========================================
        self.pub_left_delta = self.create_publisher(Twist, '/fr5_left/action_delta', 10)
        self.pub_left_grip = self.create_publisher(Float32, '/fr5_left/desired_gripper_pose', 10)
        
        self.pub_right_delta = self.create_publisher(Twist, '/fr5_right/action_delta', 10)
        self.pub_right_grip = self.create_publisher(Float32, '/fr5_right/desired_gripper_pose', 10)

        self.pub_needle_pts = self.create_publisher(Float32MultiArray, '/clicked_needle_points', 10)
        self.needle_pub_timer = self.create_timer(0.1, self.publish_needle_points_callback)

        self.pub_image1_inference = self.create_publisher(Image, '/image1_inference', 10)

        # ==========================================
        # 5. Service Clients
        # ==========================================
        self.depth_cli = self.create_client(RelativeDepthMapService, 'relative_depth_inference', callback_group=self.callback_group)
        
        self.inference_thread = threading.Thread(target=self.run_inference_loop, daemon=True)

    # --- Callbacks ---
    def task_callback(self, msg): self.latest_data['task_string'] = msg.data
    def update_joint(self, key, msg): self.latest_data[key] = np.array(msg.position, dtype=np.float32)
    def update_scalar(self, key, msg): self.latest_data[key] = np.array([msg.data], dtype=np.float32)
    def update_pose(self, key, msg):
        arr = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w], dtype=np.float32)
        self.latest_data[key] = arr
    def image_callback(self, msg):
        try: self.latest_data['image1'] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e: self.get_logger().error(f"Image decode fail: {e}")

    def publish_needle_points_callback(self):
        msg = Float32MultiArray()
        msg.data = [float(x) for x in self.latest_data['needle_location_vec']]
        self.pub_needle_pts.publish(msg)

    # --- UI Logic ---
    def mouse_callback(self, event, x, y, flags, param):
        if self.is_inference_running:
            return 
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.clicked_points) < 2:
                self.clicked_points.append((x, y))
                pt_type = "Insertion (Green)" if len(self.clicked_points) == 1 else "Exit (Red)"
                self.get_logger().info(f"Selected {pt_type} at Zoomed Scale: ({x}, {y})")

    def run_gui_loop(self):
        window_name = "Suturing Controller (2x View)"
        infer_window_name = "Inference View (Original Size Afterimage)"
        
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        self.get_logger().info("GUI Started. Select 2 points and press Enter.")

        while rclpy.ok():
            if self.latest_data['image1'] is None:
                time.sleep(0.1)
                continue

            raw_img = self.latest_data['image1'].copy()
            display_img = cv2.resize(raw_img, None, fx=self.DISPLAY_SCALE, fy=self.DISPLAY_SCALE, interpolation=cv2.INTER_LINEAR)

            for i, pt in enumerate(self.clicked_points):
                color = (0, 255, 0) if i == 0 else (255, 0, 0) 
                cv2.circle(display_img, pt, 6, color, -1)
                cv2.putText(display_img, str(i+1), (pt[0]+15, pt[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if self.is_inference_running:
                status_text = "RUNNING (Press 'R' to Stop)"
                color = (0, 0, 255)
            else:
                if len(self.clicked_points) == 2:
                    status_text = "READY (Press 'Enter')"
                    color = (255, 0, 0) 
                else:
                    status_text = f"Select Points ({len(self.clicked_points)}/2)"
                    color = (0, 255, 255)

            cv2.putText(display_img, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            current_task = self.latest_data['task_string']
            task_text = f"Task: {current_task}" if current_task else "Task: Waiting..."
            cv2.putText(display_img, task_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # 1. 메인 컨트롤러 창 표시
            cv2.imshow(window_name, display_img)
            
            # 2. [NEW] 인퍼런스 실시간 잔상 이미지 표시 (실행 중일 때만)
            if self.is_inference_running and self.current_afterimage_display is not None:
                cv2.imshow(infer_window_name, self.current_afterimage_display)

            key = cv2.waitKey(20) & 0xFF

            if key == 13: # Enter
                if len(self.clicked_points) == 2 and not self.is_inference_running:
                    self.update_needle_vector() 
                    self.is_inference_running = True
                    self.get_logger().info(">>> INFERENCE STARTED <<<")
            elif key == ord('r') or key == ord('R'):
                self.is_inference_running = False
                self.clicked_points = []
                self.latest_data['needle_location_vec'] = np.zeros(4, dtype=np.float32)
                self.get_logger().info(">>> STOPPED & RESET <<<")
                
                # [NEW] 정지 시 인퍼런스 창 닫기
                try:
                    cv2.destroyWindow(infer_window_name)
                except cv2.error:
                    pass

            elif key == 27: # ESC
                break
                
        cv2.destroyAllWindows() # 종료 시 모든 창 깔끔하게 닫기

    def update_needle_vector(self):
        if len(self.clicked_points) == 2:
            p1 = self.clicked_points[0]
            p2 = self.clicked_points[1]
            width_scaled = self.IMG_W * self.DISPLAY_SCALE
            height_scaled = self.IMG_H * self.DISPLAY_SCALE
            norm_vec = np.array([
                p1[0] / width_scaled,  p1[1] / height_scaled,
                p2[0] / width_scaled,  p2[1] / height_scaled
            ], dtype=np.float32)
            self.latest_data['needle_location_vec'] = norm_vec
            self.get_logger().info(f"Needle Vector Updated (Normalized): {norm_vec}")

    # --- Helper Methods ---
    def call_depth_service(self, cv_image):
        if not self.depth_cli.service_is_ready():
            if not self.depth_cli.wait_for_service(timeout_sec=1.0):
                raise RuntimeError("Depth service not ready")
        req = RelativeDepthMapService.Request()
        req.image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        future = self.depth_cli.call_async(req)
        while rclpy.ok() and not future.done():
            time.sleep(0.01) 
        result = future.result()
        if result is not None:
            return self.bridge.imgmsg_to_cv2(result.depth_image, desired_encoding='passthrough')
        else:
            self.get_logger().error("Service call returned None result")
            return None

    def apply_visual_prompt(self, img_bgr, needle_vec):
        prompted_img = img_bgr.copy()
        if np.allclose(needle_vec, 0): return prompted_img
        ins_x = int(needle_vec[0] * self.IMG_W)
        ins_y = int(needle_vec[1] * self.IMG_H)
        ext_x = int(needle_vec[2] * self.IMG_W)
        ext_y = int(needle_vec[3] * self.IMG_H)
        if ins_x > 0: cv2.circle(prompted_img, (ins_x, ins_y), 6, (0, 255, 0), -1)
        if ext_x > 0: cv2.circle(prompted_img, (ext_x, ext_y), 6, (255, 0, 0), -1)
        return prompted_img

    def get_formatted_observation(self, visual_prompt_img, depth_img, task_str):
        obs = {}
        obs['image1'] = visual_prompt_img 
        obs['relative_depth_map'] = depth_img
        task_onehot = np.zeros(5, dtype=np.float32)
        idx = self.task_map.get(task_str, 0) 
        task_onehot[idx] = 1.0
        obs['task_onehot'] = task_onehot
        keys_to_copy = [
            'fr5_left_joint_states', 'fr5_left_gripper_state', 
            'fr5_left_gripper_wrt_rcm', 'fr5_left_gripper_wrt_right_gripper',
            'fr5_right_joint_states', 'fr5_right_gripper_state', 
            'fr5_right_gripper_wrt_rcm', 'fr5_right_gripper_wrt_left_gripper',
            'needle_location_vec'
        ]
        for k in keys_to_copy:
            obs[k] = self.latest_data[k]
        return obs
    
    def convert_action_to_twist(self, action_sub_vec):
        t = Twist()
        t.linear.x = float(action_sub_vec[0])
        t.linear.y = float(action_sub_vec[1])
        t.linear.z = float(action_sub_vec[2])
        t.angular.x = float(action_sub_vec[3]) 
        t.angular.y = 0.0 
        t.angular.z = 0.0
        return t

    def apply_gripper_logic(self, raw_pred_val, last_val, last_time):
        target_val = 5.0 if raw_pred_val < 0.5 else 100.0
        curr_time = time.time()
        time_diff = curr_time - last_time

        if time_diff < self.GRIP_LOCK_DURATION:
            return last_val, last_val, last_time
        
        if target_val != last_val:
            return target_val, target_val, curr_time
        else:
            return target_val, target_val, last_time

    # --- Inference Loop (Background) ---
    def run_inference_loop(self):
        self.get_logger().info("Inference Thread Initialized. Waiting for Start Flag...")
        rate_period = 1.0 / self.control_freq

        ALPHA = 0.3
        # [NEW] 리사이즈 전 원본 크기(640x480)의 잔상 버퍼
        prev_afterimage_full = None

        while rclpy.ok():
            # 1. Check Flag & Data
            if not self.is_inference_running or self.latest_data['image1'] is None:
                prev_afterimage_full = None 
                self.current_afterimage_display = None # [NEW] 초기화
                time.sleep(0.1)
                continue

            # 2. Check all required data available
            not_available_fileds = [k for k, v in self.latest_data.items() if v is None]
            if len(not_available_fileds) > 0:
                if self.latest_data['task_string'] is None:
                    try:
                        img_msg = self.bridge.cv2_to_imgmsg(self.latest_data['image1'], encoding='bgr8')
                        self.pub_image1_inference.publish(img_msg)
                    except Exception as e:
                        pass
                
                self.get_logger().warning(f"Waiting for Data: {not_available_fileds}")
                time.sleep(0.1)
                continue

            # ==============================
            # A. Prepare Observation & Inference
            # ==============================
            try:
                img_clean = self.latest_data['image1'].copy()
                current_task = self.latest_data['task_string']
                needle_vec = self.latest_data['needle_location_vec'] 

                try:
                    img_msg = self.bridge.cv2_to_imgmsg(img_clean, encoding='bgr8')
                    self.pub_image1_inference.publish(img_msg)
                except Exception as e:
                    pass

                current_depth = self.call_depth_service(img_clean)
                
                # 원본 크기(640x480)의 프롬프트 이미지
                img_prompted = self.apply_visual_prompt(img_clean, needle_vec)

                # --- [NEW] 원본 크기 상태에서 잔상 효과 (알파 블렌딩) 적용 ---
                if prev_afterimage_full is None:
                    img_prompted_blended = img_prompted.copy()
                else:
                    img_prompted_blended = cv2.addWeighted(img_prompted, ALPHA, prev_afterimage_full, 1 - ALPHA, 0)
                
                # 누적 및 GUI 표시 변수 업데이트
                prev_afterimage_full = img_prompted_blended
                self.current_afterimage_display = img_prompted_blended.copy() # GUI 스레드로 전달

                # 인퍼런스를 위해 블렌딩 된 이미지를 리사이즈 (320x240)
                img_final_blended = cv2.resize(img_prompted_blended, (self.TARGET_W, self.TARGET_H), interpolation=cv2.INTER_AREA)
                depth_final = cv2.resize(current_depth, (self.TARGET_W, self.TARGET_H), interpolation=cv2.INTER_NEAREST)

                obs_dict = self.get_formatted_observation(img_final_blended, depth_final, current_task)
                self.obs_buffer.append(obs_dict)

                while len(self.obs_buffer) < self.n_obs_steps:
                    self.obs_buffer.append(obs_dict)

                payload = self.prepare_payload(self.obs_buffer)

                starttime = time.time()
                response = requests.post(self.inference_url, json=payload)
                response.raise_for_status()
                
                actions_seq = np.array(response.json()) 
                if len(actions_seq.shape) == 1: actions_seq = actions_seq[np.newaxis, :]

                # ==============================
                # B. Execute Action Chunk
                # ==============================
                for i in range(len(actions_seq)):
                    step_start = time.time()
                    if not self.is_inference_running: break

                    action_vec = actions_seq[i] 
                    
                    # --- Left Arm ---
                    l_twist = self.convert_action_to_twist(action_vec[:4]) 
                    raw_l_grip = action_vec[4]
                    prev_l_val = self.last_left_grip_val
                    prev_r_val = self.last_right_grip_val

                    final_l_grip, self.last_left_grip_val, self.last_left_grip_time = \
                        self.apply_gripper_logic(raw_l_grip, self.last_left_grip_val, self.last_left_grip_time)

                    # --- Right Arm ---
                    r_twist = self.convert_action_to_twist(action_vec[5:9])
                    raw_r_grip = action_vec[9]

                    final_r_grip, self.last_right_grip_val, self.last_right_grip_time = \
                        self.apply_gripper_logic(raw_r_grip, self.last_right_grip_val, self.last_right_grip_time)
                    
                    if final_l_grip != prev_l_val or final_r_grip != prev_r_val:
                        self.get_logger().info(f"Gripper Action Detected! Freezing Arm for {self.ARM_PAUSE_DURATION}s")
                        self.arm_stop_until = time.time() + self.ARM_PAUSE_DURATION

                    if time.time() < self.arm_stop_until:
                        l_twist = Twist() 
                        r_twist = Twist() 

                    self.pub_left_delta.publish(l_twist)
                    self.pub_left_grip.publish(Float32(data=final_l_grip))

                    self.pub_right_delta.publish(r_twist)
                    self.pub_right_grip.publish(Float32(data=final_r_grip))

                    # --- Loop Control ---
                    elapsed = time.time() - step_start
                    sleep_time = rate_period - elapsed
                    if sleep_time > 0: time.sleep(sleep_time)
                    
                    # Buffer Update
                    new_img = self.latest_data['image1'].copy()
                    
                    try:
                        new_img_msg = self.bridge.cv2_to_imgmsg(new_img, encoding='bgr8')
                        self.pub_image1_inference.publish(new_img_msg)
                    except Exception as e:
                        pass

                    new_depth = self.call_depth_service(new_img) 
                    
                    # 원본 크기(640x480)의 프롬프트 이미지
                    new_prompted = self.apply_visual_prompt(new_img, needle_vec)

                    # --- [NEW] 원본 크기 상태에서 잔상 효과 (알파 블렌딩) 적용 ---
                    if prev_afterimage_full is None:
                        new_prompted_blended = new_prompted.copy()
                    else:
                        new_prompted_blended = cv2.addWeighted(new_prompted, ALPHA, prev_afterimage_full, 1 - ALPHA, 0)
                    
                    # 누적 및 GUI 표시 변수 업데이트
                    prev_afterimage_full = new_prompted_blended
                    self.current_afterimage_display = new_prompted_blended.copy() # GUI 스레드로 전달

                    # 인퍼런스를 위해 블렌딩 된 이미지를 리사이즈 (320x240)
                    new_final_blended = cv2.resize(new_prompted_blended, (self.TARGET_W, self.TARGET_H), interpolation=cv2.INTER_AREA)
                    new_depth_final = cv2.resize(new_depth, (self.TARGET_W, self.TARGET_H), interpolation=cv2.INTER_NEAREST)

                    new_obs = self.get_formatted_observation(new_final_blended, new_depth_final, current_task)
                    self.obs_buffer.append(new_obs)

            except Exception as e:
                self.get_logger().error(f"Inference Loop Error: {e}")
                time.sleep(0.1)

    def prepare_payload(self, buffer):
        payload = {
            'image1': [], 'relative_depth_map': [], 'task_onehot': [],
            'fr5_left_joint_states': [], 'fr5_left_gripper_state': [],
            'fr5_left_gripper_wrt_rcm': [], 'fr5_left_gripper_wrt_right_gripper': [],
            'fr5_right_joint_states': [], 'fr5_right_gripper_state': [],
            'fr5_right_gripper_wrt_rcm': [], 'fr5_right_gripper_wrt_left_gripper': [],
            'needle_location_vec': []
        }
        for obs in buffer:
            _, img_enc = cv2.imencode('.jpg', obs['image1'])
            payload['image1'].append(base64.b64encode(img_enc).decode('utf-8'))
            _, depth_enc = cv2.imencode('.png', obs['relative_depth_map'])
            payload['relative_depth_map'].append(base64.b64encode(depth_enc).decode('utf-8'))
            for k in payload.keys():
                if k not in ['image1', 'relative_depth_map']:
                    payload[k].append(obs[k].tolist())
        return payload

def main(args=None):
    rclpy.init(args=args)
    node = SuturingMasterNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    node.inference_thread.start()
    try:
        node.run_gui_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()