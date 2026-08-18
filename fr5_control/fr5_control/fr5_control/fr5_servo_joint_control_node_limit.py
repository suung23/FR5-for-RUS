import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from fr5_control.fairino import Robot
import math
from geometry_msgs.msg import Pose, TransformStamped
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster
import time
import threading
import signal
import sys

# External Library Imports
from scipy.spatial.transform import Rotation 

J6_LIMIT_DEG = 165


class FR5ServoJointControlNode(Node):
    def __init__(self, robot_name, robot_ip):
        # Initialize with specific name
        super().__init__(f'{robot_name}_servo_joint_control_node')
        
        self.robot_name = robot_name
        
        # 1. Parameters
        self.declare_parameter('cmd_t', 0.008)
        self.declare_parameter('max_joint_vel', 1.5) 
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'tool_center_point')
        
        self.cmd_t = self.get_parameter('cmd_t').get_parameter_value().double_value
        self.max_vel = self.get_parameter('max_joint_vel').get_parameter_value().double_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.tool_frame = self.get_parameter('tool_frame').get_parameter_value().string_value
        
        # 2. Connection
        try:
            self.get_logger().info(f"Connecting to {self.robot_name} at {robot_ip}...")
            self.robot = Robot.RPC(robot_ip)
            robot_state = self.robot.robot_state_pkg
            self.get_logger().info(f"{self.robot_name} connected")
            
            self.current_joint_pos_deg = list(robot_state.jt_cur_pos)
            
            # --- [Homing Logic 1] 초기 위치 저장 ---
            # list()로 복사하여 참조가 아닌 값으로 저장합니다.
            self.initial_home_pos_deg = list(self.current_joint_pos_deg)
            self.get_logger().info(f"{self.robot_name} Home Position Saved: {self.initial_home_pos_deg}")
            # -------------------------------------

            self.robot.ServoMoveStart()

            self.get_logger().info(f"{self.robot_name} resetting gripper")
            self.robot.ActGripper(1,0)
            time.sleep(1)
            self.get_logger().info(f"{self.robot_name} Activating gripper") 
            self.robot.ActGripper(1,1)
            time.sleep(1)
            self.get_logger().info(f'{self.robot_name} Servo Mode Started with Velocity Limiter ({self.max_vel} rad/s)')
        except Exception as e:
            self.get_logger().error(f'{self.robot_name} Connection Error: {e}')
            return
        
        self.get_logger().info(f"{self.robot_name} robot_state: {self.robot.robot_state_pkg.robot_state}")


        # 3. Variables
        self.target_vel_rad = [0.0] * 6
        self.last_msg_time = self.get_clock().now()
        self.epos = [0.0] * 4  
        self.cmd_id = 0
        self.vel = 0.0
        self.acc = 0.0
        self.filterT = 0.0
        self.gain = 0.0
        
        self.last_control_time = self.get_clock().now()

        # 4. Subscriber
        self.velocity_sub = self.create_subscription(
            JointState,
            f'/{self.robot_name}/joint_velocity_cmds',
            self.velocity_callback,
            10)
        
        self.gripper_sub = self.create_subscription(
            Float32,
            f'/{self.robot_name}/desired_gripper_pose',
            self.gripper_callback,
            10
        )

        self.last_gripper_state = 100.0

        # 5. Timer for control loop
        self.timer = self.create_timer(self.cmd_t, self.control_loop)

        ## 6 for status
        self.joint_pub = self.create_publisher(JointState, f'/{self.robot_name}/joint_states', 10)
        self.pose_pub = self.create_publisher(Pose, f'/{self.robot_name}/ee_wrt_base', 10)
        self.gripper_pub = self.create_publisher(Float32, f'/{self.robot_name}/gripper_state', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.status_timer = self.create_timer(0.0166, self.status_timer_callback)
        
        self.get_logger().info(f"{self.robot_name} Testing gripper...")

        # test out gripers
        self.gripper_callback(Float32(data=5.0)) 
        time.sleep(2)
        self.gripper_callback(Float32(data=100.0)) 
        time.sleep(2)

        self.get_logger().info(f"{self.robot_name} Testing complete! Good to go")

    # --- [Homing Logic 2] 호밍 함수 구현 ---
    def perform_homing(self):
        """
        서보 모드를 종료하고 초기 위치로 천천히 이동합니다.
        """
        try:
            self.get_logger().info(f"[{self.robot_name}] Stopping Servo Mode for Homing...")
            self.robot.ServoMoveEnd()
            time.sleep(0.5) 

            self.get_logger().info(f"[{self.robot_name}] Moving to Start Position (Speed: 15%)...")
            
            # --- [수정된 부분] ---
            # 문서에 따른 MoveJ 파라미터 적용
            # 1. joint_pos: 초기 관절 각도
            # 2. tool, user: 0
            # 3. vel: 15.0 (속도 15%)
            # 4. blendT: -1.0 (Blocking, 해당 지점에 완전히 멈춤)
            # 나머지(desc_pos 등)는 Default 값을 사용하도록 키워드 인자(vel=...) 사용
            
            ret = self.robot.MoveJ(
                joint_pos=self.initial_home_pos_deg, 
                tool=0, 
                user=0, 
                vel=15.0,     # desc_pos를 건너뛰고 vel을 명시적으로 지정
                blendT=-1.0   # 문서상 -1.0은 motion in place (blocking)
            )
            # --------------------
            
            if ret != 0:
                self.get_logger().error(f"[{self.robot_name}] MoveJ Failed with error code: {ret}")
            else:
                self.get_logger().info(f"[{self.robot_name}] Homing command sent. Waiting for completion...")
                # blendT가 -1.0이어도 RPC 특성상 명령만 보내고 바로 리턴될 수 있으므로
                # 물리적인 이동 시간을 위해 sleep을 유지하는 것이 안전합니다.
                time.sleep(4.0) 
                self.get_logger().info(f"[{self.robot_name}] Homing Complete.")

        except Exception as e:
            self.get_logger().error(f"[{self.robot_name}] Homing Exception: {e}")
    # -------------------------------------

    def gripper_callback(self, msg: Float32):
        desired_pose = float(msg.data)
        if desired_pose != self.last_gripper_state:
            self.last_gripper_state = desired_pose
            self.robot.ServoMoveEnd()
            self.get_logger().info(f"{self.robot_name} Moving gripper...")

            self.robot.MoveGripper(1, desired_pose, 100, 50, 3000, 0, 0, 0, 0, 0)
            self.get_logger().info(f"{self.robot_name} Gripper moved at position {desired_pose}.")
            
            if self.robot.robot_state_pkg:
                self.current_joint_pos_deg = list(self.robot.robot_state_pkg.jt_cur_pos)
            
            self.target_vel_rad = [0.0] * 6
            self.last_control_time = self.get_clock().now()
            self.robot.ServoMoveStart() 

    def velocity_callback(self, msg):
        now = self.get_clock().now()
        try:
            msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
            lag = (now - msg_time).nanoseconds / 1e9
            if lag > 0.5:
                return
        except Exception:
            pass

        if len(msg.velocity) >= 6:
            self.target_vel_rad = list(msg.velocity)
            self.last_msg_time = now

    def control_loop(self):
        try:
            self.cmd_id += 1
            now = self.get_clock().now()
            dt = (now - self.last_control_time).nanoseconds / 1e9
            self.last_control_time = now

            if dt > 0.1:
                dt = self.cmd_t 
            elif dt < 0.002:
                dt = 0.002
            
            time_since_last_msg = (now - self.last_msg_time).nanoseconds / 1e9
            if time_since_last_msg > 0.5:
                self.target_vel_rad = [0.0] * 6

            for i in range(6):
                clamped_vel = max(min(self.target_vel_rad[i], self.max_vel), -self.max_vel)
                vel_deg_s = math.degrees(clamped_vel)
                self.current_joint_pos_deg[i] += vel_deg_s * dt
                
                
            # apply clipping on j6
            self.current_joint_pos_deg[5] = max(min(self.current_joint_pos_deg[5], J6_LIMIT_DEG), -J6_LIMIT_DEG)

            error = self.robot.ServoJ(
                joint_pos=self.current_joint_pos_deg,
                axisPos=self.epos,
                acc=self.acc,
                vel=self.vel, 
                cmdT=dt, 
                filterT=self.filterT,
                gain=self.gain, 
                id=self.cmd_id,
            )

            if error != 0:
                self.get_logger().error(f"{self.robot_name} ServoJ Error: {error}", throttle_duration_sec=1.0)
            
        except Exception as e:
            self.get_logger().error(f"{self.robot_name} ServoJ control loop Error: {e}", throttle_duration_sec=1.0)

    def status_timer_callback(self):
        try:
            state = self.robot.robot_state_pkg
            now = self.get_clock().now().to_msg()

            jointstate_msg = JointState()
            jointstate_msg.header.stamp = now
            jointstate_msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            jointstate_msg.position = [math.radians(deg) for deg in state.jt_cur_pos]
            jointstate_msg.velocity = [math.radians(deg_s) for deg_s in state.actual_qd]
            jointstate_msg.effort = list(state.jt_cur_tor)
            self.joint_pub.publish(jointstate_msg)
            
            x, y, z = [val / 1000.0 for val in state.tl_cur_pos[0:3]]
            euler_deg = [state.tl_cur_pos[3], state.tl_cur_pos[4], state.tl_cur_pos[5]]
            rot = Rotation.from_euler('xyz', euler_deg, degrees=True)
            quat = rot.as_quat() 
            pose_msg = Pose()
            pose_msg.position.x = x
            pose_msg.position.y = y
            pose_msg.position.z = z
            pose_msg.orientation.x = quat[0]
            pose_msg.orientation.y = quat[1]
            pose_msg.orientation.z = quat[2]
            pose_msg.orientation.w = quat[3]
            self.pose_pub.publish(pose_msg)

            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = f'{self.robot_name}_{self.base_frame}' 
            t.child_frame_id = f'{self.robot_name}_{self.tool_frame}' 
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]
            self.tf_broadcaster.sendTransform(t)

            ret = self.robot.GetGripperCurPosition()
            if isinstance(ret, int):
                err_code = ret
                fault = 0
                position = 0
            else:
                err_code, fault, position = ret

            if err_code != 0:
                pass
            else:
                if position != 0:
                    self.last_gripper_state = position 
                if position == 0:
                    position = self.last_gripper_state 
                gripper_msg = Float32()
                gripper_msg.data = float(position)
                self.gripper_pub.publish(gripper_msg)

        except Exception as e:
            self.get_logger().warn(f'Error reading robot state: {e}')

    def destroy_node(self):
        # destroy_node에서는 이제 연결만 끊습니다.
        try:
            # Homing은 main에서 명시적으로 호출합니다.
            self.robot.ServoMoveEnd()
            self.robot.ActGripper(1,0)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    # Instantiate two nodes
    left_node = FR5ServoJointControlNode('fr5_left', '192.168.58.2')
    right_node = FR5ServoJointControlNode('fr5_right', '192.168.58.3')
    
    executor = MultiThreadedExecutor()
    executor.add_node(left_node)
    executor.add_node(right_node)
    
    def signal_handler(sig, frame):
        # 종료 시그널을 받으면 KeyboardInterrupt 발생
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, signal_handler)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down detected! Starting Homing Sequence...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # 1. Executor 중지 (Control loop 중단)
        if executor:
            executor.shutdown()
        
        # --- [수정된 부분: 병렬 호밍 실행] ---
        print("Starting parallel homing...")

        # 두 개의 스레드 생성
        t_left = threading.Thread(target=left_node.perform_homing)
        t_right = threading.Thread(target=right_node.perform_homing)

        # 스레드 시작 (동시에 로봇 움직임 시작)
        t_left.start()
        t_right.start()

        # 두 스레드가 끝날 때까지 메인 스레드 대기 (4초 이상 대기하게 됨)
        t_left.join()
        t_right.join()
        
        print("Both robots homing sequence finished.")
        # -------------------------------------
        
        try:
            left_node.destroy_node()
            right_node.destroy_node()
        except:
            pass
            
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()