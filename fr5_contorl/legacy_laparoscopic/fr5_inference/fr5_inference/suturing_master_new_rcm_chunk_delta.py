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
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String, Float32MultiArray
from geometry_msgs.msg import Twist 

# Service Imports
from suturing_interfaces.srv import RelativeDepthMapService

class SuturingMasterNode(Node):
    def __init__(self):
        super().__init__('suturing_master_node')
        
        # ==========================================
        # 1. 파라미터 및 설정
        # ==========================================
        self.declare_parameter('control_frequency', 20.0) 
        self.declare_parameter('n_obs_steps', 2)          
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
        # 그리퍼 상태 관리 변수
        # ==========================================
        self.GRIP_LOCK_DURATION = 3.0  
        self.ARM_PAUSE_DURATION = 2.0  
        self.arm_stop_until = 0.0      

        # === [그리퍼 연속 N번 로직 추가] ===
        self.GRIP_CONSECUTIVE_THRESHOLD = 3  # 연속 3번 같은 상태가 예측되어야 실제로 변경 (N값)

        self.last_left_grip_val = 100.0  
        self.last_left_grip_time = 0.0   
        self.left_grip_candidate_val = 100.0
        self.left_grip_consecutive_count = 0

        self.last_right_grip_val = 100.0 
        self.last_right_grip_time = 0.0  
        self.right_grip_candidate_val = 100.0
        self.right_grip_consecutive_count = 0
        # ====================================

        # ==========================================
        # 2. 데이터 버퍼 및 상태 변수
        # ==========================================
        self.latest_data = {
            "task_string": "Needle_Pickup",          
            "image1": None,               
            "fr5_left_gripper_state": None,
            "fr5_left_gripper_wrt_new_rcm": None,
            "fr5_right_gripper_state": None,
            "fr5_right_gripper_wrt_new_rcm": None,
            "needle_location_vec": np.zeros(4, dtype=np.float32) 
        }

        self.obs_buffer = deque(maxlen=self.n_obs_steps)

        self.clicked_points = [] 
        self.is_inference_running = False 
        
        self.latest_inference_time = 0.0

        # ==========================================
        # 3. Subscribers
        # ==========================================
        self.create_subscription(String, '/predicted_task', self.task_callback, 10, callback_group=self.callback_group)
        self.create_subscription(Image, '/camera/image1', self.image_callback, 10, callback_group=self.callback_group)
        
        self.create_subscription(Float32, '/fr5_left/gripper_state', lambda m: self.update_scalar('fr5_left_gripper_state', m), 10, callback_group=self.callback_group)
        self.create_subscription(Float32MultiArray, '/fr5_left/gripper_wrt_new_rcm', lambda m: self.update_4d_state('fr5_left_gripper_wrt_new_rcm', m), 10, callback_group=self.callback_group)
        
        self.create_subscription(Float32, '/fr5_right/gripper_state', lambda m: self.update_scalar('fr5_right_gripper_state', m), 10, callback_group=self.callback_group)
        self.create_subscription(Float32MultiArray, '/fr5_right/gripper_wrt_new_rcm', lambda m: self.update_4d_state('fr5_right_gripper_wrt_new_rcm', m), 10, callback_group=self.callback_group)

        # ==========================================
        # 4. Publishers
        # ==========================================
        self.pub_left_abs = self.create_publisher(Twist, '/fr5_left/action_absolute', 10)
        self.pub_left_grip = self.create_publisher(Float32, '/fr5_left/desired_gripper_pose', 10)
        
        self.pub_right_abs = self.create_publisher(Twist, '/fr5_right/action_absolute', 10)
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
    def update_scalar(self, key, msg): self.latest_data[key] = np.array([msg.data], dtype=np.float32)
    def image_callback(self, msg):
        try: self.latest_data['image1'] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e: self.get_logger().error(f"Image decode fail: {e}")

    def update_4d_state(self, key, msg):
        arr = np.array(msg.data, dtype=np.float32)
        self.latest_data[key] = arr

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
            task_text = f"Predicted Task: {current_task}" if current_task else "Predicted Task: None"
            cv2.putText(display_img, task_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            inf_time_text = f"Inf Time: {self.latest_inference_time:.3f} s"
            cv2.putText(display_img, inf_time_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 255), 2)

            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(20) & 0xFF

            if key == 13: # Enter
                if len(self.clicked_points) == 2 and not self.is_inference_running:
                    self.update_needle_vector() 
                    self.is_inference_running = True
                    self.latest_inference_time = 0.0
                    
                    # 루프 재시작 시 카운트 리셋
                    self.left_grip_consecutive_count = 0
                    self.right_grip_consecutive_count = 0
                    
                    self.get_logger().info(">>> INFERENCE STARTED <<<")
            elif key == ord('r') or key == ord('R'):
                self.is_inference_running = False
                self.clicked_points = []
                self.latest_data['needle_location_vec'] = np.zeros(4, dtype=np.float32)
                self.latest_inference_time = 0.0
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
            'fr5_left_gripper_wrt_new_rcm', 'fr5_left_gripper_state',
            'fr5_right_gripper_wrt_new_rcm', 'fr5_right_gripper_state',
            'needle_location_vec'
        ]
        for k in keys_to_copy:
            obs[k] = self.latest_data[k]
        return obs
    
    def convert_absolute_action_to_twist(self, action_sub_vec):
        t = Twist()
        t.linear.x = float(action_sub_vec[0]) 
        t.linear.y = float(action_sub_vec[1]) 
        t.linear.z = float(action_sub_vec[2]) 
        t.angular.x = float(action_sub_vec[3]) 
        t.angular.y = 0.0 
        t.angular.z = 0.0
        return t

    # === [그리퍼 연속 N번 로직 추가] 변경된 함수 ===
    def apply_gripper_logic(self, raw_pred_val, current_actual_val, last_action_time, candidate_val, consecutive_count):
        """
        raw_pred_val: 모델의 날 예측값
        current_actual_val: 현재 실제로 그리퍼에 인가된 타겟값 (100.0 or 5.0)
        last_action_time: 마지막으로 상태를 변경한 시간
        candidate_val: 현재 누적 중인 변경 후보값
        consecutive_count: 후보값이 연속으로 나온 횟수
        """
        desired_val = 5.0 if raw_pred_val < 0.5 else 100.0
        curr_time = time.time()
        time_diff = curr_time - last_action_time

        # 1. 시간 락 체킹 (락 걸려있으면 카운트는 0으로 유지)
        if time_diff < self.GRIP_LOCK_DURATION:
            return current_actual_val, current_actual_val, last_action_time, candidate_val, 0
        
        # 2. 이미 원하는 상태에 도달해 있다면 카운트 리셋 후 유지
        if desired_val == current_actual_val:
            return current_actual_val, current_actual_val, last_action_time, desired_val, 0
        
        # 3. 변경해야 하는 상태라면 카운트 누적
        if desired_val == candidate_val:
            consecutive_count += 1
        else:
            # 방향이 갑자기 바뀌면 카운트 리셋
            candidate_val = desired_val
            consecutive_count = 1
            
        # 4. 임계치(N) 도달 확인
        if consecutive_count >= self.GRIP_CONSECUTIVE_THRESHOLD:
            # 최종적으로 상태 변경 확정
            return desired_val, desired_val, curr_time, desired_val, 0
        else:
            # 아직 임계치 도달 안함. 기존 상태 유지하며 카운트만 올림
            return current_actual_val, current_actual_val, last_action_time, candidate_val, consecutive_count
    # ===============================================

    # --- Inference Loop ---
    def run_inference_loop(self):
        self.get_logger().info("Inference Thread Initialized. Waiting for Start Flag...")
        rate_period = 1.0 / self.control_freq

        while rclpy.ok():
            if not self.is_inference_running or self.latest_data['image1'] is None:
                time.sleep(0.1)
                continue

            not_available_fileds = [k for k, v in self.latest_data.items() if v is None]
            if len(not_available_fileds) > 0:
                time.sleep(0.1)
                continue

            try:
                img_clean = self.latest_data['image1'].copy()
                current_task = self.latest_data['task_string']
                needle_vec = self.latest_data['needle_location_vec'] 

                try:
                    self.pub_image1_inference.publish(self.bridge.cv2_to_imgmsg(img_clean, encoding='bgr8'))
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

                http_start_time = time.time()
                response = requests.post(self.inference_url, json=payload)
                response.raise_for_status()
                self.latest_inference_time = time.time() - http_start_time
                
                actions_seq = np.array(response.json()) # (T, 10)
                if len(actions_seq.shape) == 1: actions_seq = actions_seq[np.newaxis, :]

                # =======================================================
                # [추가됨] Chunk-wise Delta -> Absolute Action 역변환
                # =======================================================
                # 현재 로봇의 최신 상태(Observation 중 가장 마지막)를 가져옵니다.
                latest_obs = self.obs_buffer[-1]
                chunk_start = np.concatenate([
                    latest_obs['fr5_left_gripper_wrt_new_rcm'],   # 0~3
                    latest_obs['fr5_left_gripper_state'],         # 4
                    latest_obs['fr5_right_gripper_wrt_new_rcm'],  # 5~8
                    latest_obs['fr5_right_gripper_state']         # 9
                ])

                absolute_actions_seq = np.zeros_like(actions_seq)
                for step_idx in range(len(actions_seq)):
                    delta_vec = actions_seq[step_idx]
                    abs_vec = np.zeros(10, dtype=np.float32)
                    
                    # 1. 위치 및 자세 복원 (현재 기준점 + Delta)
                    abs_vec[0:4] = delta_vec[0:4] + chunk_start[0:4]
                    abs_vec[5:9] = delta_vec[5:9] + chunk_start[5:9]
                    
                    # 2. Roll 각도 -pi ~ pi 보정 (인덱스 3, 8)
                    abs_vec[3] = (abs_vec[3] + np.pi) % (2 * np.pi) - np.pi
                    abs_vec[8] = (abs_vec[8] + np.pi) % (2 * np.pi) - np.pi
                    
                    # 3. 그리퍼 (이미 모델에서 절대값으로 나오므로 덧셈하지 않고 그대로 사용)
                    abs_vec[4] = delta_vec[4]
                    abs_vec[9] = delta_vec[9]
                    
                    absolute_actions_seq[step_idx] = abs_vec
                    
                actions_seq = absolute_actions_seq
                # =======================================================

                for i in range(len(actions_seq)):
                    step_start = time.time()
                    if not self.is_inference_running: break

                    action_vec = actions_seq[i] 
                    
                    # --- Left Arm (N번 연속 체킹 적용) ---
                    l_twist = self.convert_absolute_action_to_twist(action_vec[:4]) 
                    raw_l_grip = action_vec[4]
                    
                    prev_l_val = self.last_left_grip_val
                    prev_r_val = self.last_right_grip_val

                    final_l_grip, self.last_left_grip_val, self.last_left_grip_time, self.left_grip_candidate_val, self.left_grip_consecutive_count = \
                        self.apply_gripper_logic(
                            raw_l_grip, 
                            self.last_left_grip_val, 
                            self.last_left_grip_time,
                            self.left_grip_candidate_val,
                            self.left_grip_consecutive_count
                        )

                    # --- Right Arm (N번 연속 체킹 적용) ---
                    r_twist = self.convert_absolute_action_to_twist(action_vec[5:9])
                    raw_r_grip = action_vec[9]

                    final_r_grip, self.last_right_grip_val, self.last_right_grip_time, self.right_grip_candidate_val, self.right_grip_consecutive_count = \
                        self.apply_gripper_logic(
                            raw_r_grip, 
                            self.last_right_grip_val, 
                            self.last_right_grip_time,
                            self.right_grip_candidate_val,
                            self.right_grip_consecutive_count
                        )
                    
                    if final_l_grip != prev_l_val or final_r_grip != prev_r_val:
                        self.get_logger().info(f"Gripper Action Confirmed! Freezing Arm for {self.ARM_PAUSE_DURATION}s")
                        self.arm_stop_until = time.time() + self.ARM_PAUSE_DURATION

                    if time.time() < self.arm_stop_until:
                        curr_l_state = self.latest_data['fr5_left_gripper_wrt_new_rcm']
                        curr_r_state = self.latest_data['fr5_right_gripper_wrt_new_rcm']
                        l_twist = self.convert_absolute_action_to_twist(curr_l_state)
                        r_twist = self.convert_absolute_action_to_twist(curr_r_state)

                    self.pub_left_abs.publish(l_twist)
                    self.pub_left_grip.publish(Float32(data=final_l_grip))

                    self.pub_right_abs.publish(r_twist)
                    self.pub_right_grip.publish(Float32(data=final_r_grip))

                    # --- Loop Control ---
                    elapsed = time.time() - step_start
                    sleep_time = rate_period - elapsed
                    if sleep_time > 0: time.sleep(sleep_time)
                    
                    new_img = self.latest_data['image1'].copy()

                    try:
                        self.pub_image1_inference.publish(self.bridge.cv2_to_imgmsg(new_img, encoding='bgr8'))
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
            'fr5_left_gripper_wrt_new_rcm': [], 'fr5_left_gripper_state': [],
            'fr5_right_gripper_wrt_new_rcm': [], 'fr5_right_gripper_state': [],
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