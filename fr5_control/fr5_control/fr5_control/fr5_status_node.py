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

class FR5StatusNode(Node):
    def __init__(self, robot_name, robot_ip):
        # Initialize with specific name (status_node)
        super().__init__(f'{robot_name}_status_node')
        
        self.robot_name = robot_name
        
        # 1. Parameters (제어 관련 파라미터 제거, 프레임 관련만 유지)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'tool_center_point')
        
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.tool_frame = self.get_parameter('tool_frame').get_parameter_value().string_value
        
        # 2. Connection
        try:
            self.get_logger().info(f"Connecting to {self.robot_name} at {robot_ip}...")
            # Robot RPC 연결만 수행 (ServoMoveStart, ActGripper 등 제거)
            self.robot = Robot.RPC(robot_ip)
            
            # 연결 확인을 위한 초기 상태 읽기
            if self.robot.robot_state_pkg:
                self.get_logger().info(f"{self.robot_name} connected successfully.")
                self.get_logger().info(f"{self.robot_name} initial joint pos: {list(self.robot.robot_state_pkg.jt_cur_pos)}")
            else:
                self.get_logger().warn(f"{self.robot_name} connected but state package is empty.")

        except Exception as e:
            self.get_logger().error(f'{self.robot_name} Connection Error: {e}')
            return
        
        # 3. Variables
        self.last_gripper_state = 0.0

        # 4. Publishers (기존 토픽 이름 유지)
        self.joint_pub = self.create_publisher(JointState, f'/{self.robot_name}/joint_states', 10)
        self.pose_pub = self.create_publisher(Pose, f'/{self.robot_name}/ee_wrt_base', 10)
        self.gripper_pub = self.create_publisher(Float32, f'/{self.robot_name}/gripper_state', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # 5. Timer for Status Loop
        # 30 HZ for smooth visualization (0.033s)
        self.status_timer = self.create_timer(0.033, self.status_timer_callback)

        self.get_logger().info(f"{self.robot_name} Status Node Started (Read-Only Mode)")

    def status_timer_callback(self):
        """
        로봇의 상태를 읽어 ROS Topic으로 발행하는 콜백 함수
        """
        try:
            # fairino SDK가 백그라운드에서 업데이트하는 state package 접근
            state = self.robot.robot_state_pkg
            if state is None:
                return

            now = self.get_clock().now().to_msg()

            ## 1. Joint State Publishing
            jointstate_msg = JointState()
            jointstate_msg.header.stamp = now
            jointstate_msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

            # Degrees to Radian conversion
            jointstate_msg.position = [math.radians(deg) for deg in state.jt_cur_pos]
            jointstate_msg.velocity = [math.radians(deg_s) for deg_s in state.actual_qd]
            jointstate_msg.effort = list(state.jt_cur_tor)
            self.joint_pub.publish(jointstate_msg)

            ## 2. EE Pose Publishing
            # Conversion: mm to meters
            x, y, z = [val / 1000.0 for val in state.tl_cur_pos[0:3]]
            
            # Euler(deg) to quaternion
            euler_deg = [state.tl_cur_pos[3], state.tl_cur_pos[4], state.tl_cur_pos[5]]
            rot = Rotation.from_euler('xyz', euler_deg, degrees=True)
            quat = rot.as_quat() # [x, y, z, w]
            
            pose_msg = Pose()
            pose_msg.position.x = x
            pose_msg.position.y = y
            pose_msg.position.z = z
            pose_msg.orientation.x = quat[0]
            pose_msg.orientation.y = quat[1]
            pose_msg.orientation.z = quat[2]
            pose_msg.orientation.w = quat[3]
            self.pose_pub.publish(pose_msg)

            ## 3. Broadcast Transform (TF)
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

            ## 4. Publish Gripper State
            # 그리퍼 상태 읽기는 polling 방식일 수 있음
            ret = self.robot.GetGripperCurPosition()
            if isinstance(ret, int):
                err_code = ret
                position = 0
            else:
                err_code, fault, position = ret

            if err_code != 0:
                # 에러 발생 시 로그만 남기고 무시 (빈번할 경우 주석 처리)
                # self.get_logger().warn(f'Cannot get gripper state: {err_code}')
                pass
            else:
                # 0이 들어오면 이전 값을 유지하는 로직 (기존 코드 유지)
                if position != 0:
                    self.last_gripper_state = position 
                elif position == 0:
                    position = self.last_gripper_state 
                
                gripper_msg = Float32()
                gripper_msg.data = float(position)
                self.gripper_pub.publish(gripper_msg)

        except Exception as e:
            self.get_logger().warn(f'Error reading robot state: {e}')

    def destroy_node(self):
        # 제어 종료 명령(ServoMoveEnd 등) 제거하고 연결만 종료된다고 가정
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    # Instantiate two Status Nodes (기존 이름 및 IP 유지)
    left_node = FR5StatusNode('fr5_left', '192.168.58.2')
    right_node = FR5StatusNode('fr5_right', '192.168.58.3')
    
    executor = MultiThreadedExecutor()
    executor.add_node(left_node)
    executor.add_node(right_node)
    
    # 종료 신호(SIGTERM) 처리 핸들러
    def signal_handler(sig, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, signal_handler)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if executor:
            executor.shutdown()
        
        try:
            left_node.destroy_node()
            right_node.destroy_node()
        except:
            pass
            
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()