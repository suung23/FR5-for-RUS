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
# [수정됨] Float32MultiArray 임포트 추가
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
        self.declare_parameter('n_obs_steps', 4)          
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
        
        # [NEW] 그리퍼 작동 시 팔 정지(Freeze) 기능 추가
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

        # [추가됨] 클릭한 Needle Points를 다른 노드에서도 볼 수 있도록 퍼블리시
        self.pub_needle_pts = self.create_publisher(Float32MultiArray, '/clicked_needle_points', 10)
        # 10Hz 주기로 지속적으로 퍼블리시 (0.1초)
        self.needle_pub_timer = self.create_timer(0.1, self.publish_needle_points_callback)

        # [NEW 추가됨] 인퍼런스용 이미지를 퍼블리시하기 위한 퍼블리셔
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

    # [추가됨] Needle Points 퍼블리시용 콜백
    def publish_needle_points_callback(self):
        msg = Float32MultiArray()
        # 저장되어 있는 needle_location_vec [ins_x, ins_y, ext_x, ext_y] 를 list 형태로 퍼블리시
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

            cv2.imshow(window_name, display_img)
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
            elif key == 27: # ESC
                break
        cv2.destroyWindow(window_name)

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
        """ Twist 변환 함수 (Grip은 별도 로직으로 처리) """
        t = Twist()
        t.linear.x = float(action_sub_vec[0])
        t.linear.y = float(action_sub_vec[1])
        t.linear.z = float(action_sub_vec[2])
        t.angular.x = float(action_sub_vec[3]) # Roll Delta
        t.angular.y = 0.0 
        t.angular.z = 0.0
        return t

    def apply_gripper_logic(self, raw_pred_val, last_val, last_time):
        """
        [NEW] 그리퍼 로직:
        1. 0.5 기준으로 5.0(Close) 또는 100.0(Open) 결정
        2. 상태가 변경된지 3초가 지나지 않았으면 이전 상태 강제 유지
        """
        # 1. Thresholding
        target_val = 5.0 if raw_pred_val < 0.5 else 100.0
        
        curr_time = time.time()
        time_diff = curr_time - last_time

        # 2. Lock Logic
        if time_diff < self.GRIP_LOCK_DURATION:
            # 락 걸려있음 -> 변경 불가, 이전 값 리턴
            # last_time은 업데이트 하지 않음 (락 시작 시점 유지)
            return last_val, last_val, last_time
        
        # 락 풀림
        if target_val != last_val:
            # 상태 변경 -> 새로운 값 리턴, 시간 업데이트(락 시작)
            return target_val, target_val, curr_time
        else:
            # 상태 유지 (락은 풀려있지만 값이 같음)
            return target_val, target_val, last_time

    # --- Inference Loop (Background) ---
    def run_inference_loop(self):
        self.get_logger().info("Inference Thread Initialized. Waiting for Start Flag...")
        rate_period = 1.0 / self.control_freq

        while rclpy.ok():
            # 1. Check Flag & Data
            if not self.is_inference_running or self.latest_data['image1'] is None:
                time.sleep(0.1)
                continue

            # 2. Check all required data available
            not_available_fileds = [k for k, v in self.latest_data.items() if v is None]
            if len(not_available_fileds) > 0:
                # [NEW] 만약 task_string 정보가 없어서 대기 중이라면 계속해서 이미지를 퍼블리시
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

                # [NEW] depth service 호출 전 동일한 순간에 publish
                try:
                    img_msg = self.bridge.cv2_to_imgmsg(img_clean, encoding='bgr8')
                    self.pub_image1_inference.publish(img_msg)
                except Exception as e:
                    pass

                current_depth = self.call_depth_service(img_clean)
                img_prompted = self.apply_visual_prompt(img_clean, needle_vec)

                img_final = cv2.resize(img_prompted, (self.TARGET_W, self.TARGET_H), interpolation=cv2.INTER_AREA)
                depth_final = cv2.resize(current_depth, (self.TARGET_W, self.TARGET_H), interpolation=cv2.INTER_NEAREST)

                obs_dict = self.get_formatted_observation(img_final, depth_final, current_task)
                self.obs_buffer.append(obs_dict)

                while len(self.obs_buffer) < self.n_obs_steps:
                    self.obs_buffer.append(obs_dict)

                payload = self.prepare_payload(self.obs_buffer)

                starttime = time.time()
                response = requests.post(self.inference_url, json=payload)
                response.raise_for_status()
                
                actions_seq = np.array(response.json()) # (T, 10)
                if len(actions_seq.shape) == 1: actions_seq = actions_seq[np.newaxis, :]

                # ==============================
                # B. Execute Action Chunk
                # ==============================
                for i in range(len(actions_seq)):
                    step_start = time.time()
                    if not self.is_inference_running: break

                    action_vec = actions_seq[i] # (10,)
                    
                    # --- Left Arm ---
                    l_twist = self.convert_action_to_twist(action_vec[:4]) # dx,dy,dz,droll
                    raw_l_grip = action_vec[4]
                    
                    # [Logic] 그리퍼 값 계산 전, 이전 상태 저장 (변화 감지용)
                    prev_l_val = self.last_left_grip_val
                    prev_r_val = self.last_right_grip_val

                    # [Logic] 그리퍼 3초 락 적용 (내부에서 self.last_... 업데이트)
                    final_l_grip, self.last_left_grip_val, self.last_left_grip_time = \
                        self.apply_gripper_logic(raw_l_grip, self.last_left_grip_val, self.last_left_grip_time)

                    # --- Right Arm ---
                    r_twist = self.convert_action_to_twist(action_vec[5:9])
                    raw_r_grip = action_vec[9]

                    # [Logic] 그리퍼 3초 락 적용
                    final_r_grip, self.last_right_grip_val, self.last_right_grip_time = \
                        self.apply_gripper_logic(raw_r_grip, self.last_right_grip_val, self.last_right_grip_time)
                    
                    # [NEW] 그리퍼 상태 변화 감지 -> 팔 정지(Freeze) 트리거 설정
                    if final_l_grip != prev_l_val or final_r_grip != prev_r_val:
                        self.get_logger().info(f"Gripper Action Detected! Freezing Arm for {self.ARM_PAUSE_DURATION}s")
                        self.arm_stop_until = time.time() + self.ARM_PAUSE_DURATION

                    # [NEW] 정지 기간이라면 Twist를 0으로 덮어쓰기 (그리퍼 명령은 그대로 전송)
                    if time.time() < self.arm_stop_until:
                        l_twist = Twist() # Linear=0, Angular=0
                        r_twist = Twist() # Linear=0, Angular=0

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
                    
                    # [NEW] depth service 호출 전 동일한 순간에 publish
                    try:
                        new_img_msg = self.bridge.cv2_to_imgmsg(new_img, encoding='bgr8')
                        self.pub_image1_inference.publish(new_img_msg)
                    except Exception as e:
                        pass

                    new_depth = self.call_depth_service(new_img) 
                    new_prompted = self.apply_visual_prompt(new_img, needle_vec)
                    new_final = cv2.resize(new_prompted, (self.TARGET_W, self.TARGET_H), interpolation=cv2.INTER_AREA)
                    new_depth_final = cv2.resize(new_depth, (self.TARGET_W, self.TARGET_H), interpolation=cv2.INTER_NEAREST)
                    new_obs = self.get_formatted_observation(new_final, new_depth_final, current_task)
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