import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, TransformStamped, Twist
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster
import math
import time
import threading
import signal
import sys
import numpy as np

# Robot Interface
from fr5_control.fairino import Robot

# External Libraries
from scipy.spatial.transform import Rotation 
import PyKDL as kdl

# --- Constants ---
# J6 Joint Limit Constant
J6_LIMIT_DEG = 165.0 

CMD_DT = 0.008  # 8ms (125Hz)
FR5_BASE_DISTANCE = 0.61 

# Transformation Matrices
FR5_LEFT_GRIPPER_WRT_J6_TRANS = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.64],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

FR5_RIGHT_GRIPPER_WRT_J6_TRANS = np.array([
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.64],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

FR5_LEFT_RCM_GRIPPER_DISTANCE = 0.190
FR5_RIGHT_RCM_GRIPPER_DISTANCE = 0.192

class DualFR5IntegratedController(Node):
    def __init__(self):
        super().__init__('dual_fr5_integrated_controller')
        
        # --- Robot Configuration ---
        self.left_ip = '192.168.58.2'
        self.right_ip = '192.168.58.3'
        
        self.declare_parameter('max_joint_vel', 1.5)
        self.max_vel = self.get_parameter('max_joint_vel').get_parameter_value().double_value

        # --- Setup KDL & Transforms ---
        self.setup_kdl()

        # --- Connect to Robots ---
        self.left_robot = self.connect_robot('fr5_left', self.left_ip)
        self.right_robot = self.connect_robot('fr5_right', self.right_ip)
        
        if not self.left_robot or not self.right_robot:
            self.get_logger().error("Failed to connect to one or both robots. Exiting.")
            sys.exit(1)

        # --- Initialize Robot States (Includes Gripper Sequence 5 -> 100) ---
        self.left_state = self.init_robot_state('fr5_left', self.left_robot, FR5_LEFT_GRIPPER_WRT_J6_TRANS, FR5_LEFT_RCM_GRIPPER_DISTANCE)
        self.right_state = self.init_robot_state('fr5_right', self.right_robot, FR5_RIGHT_GRIPPER_WRT_J6_TRANS, FR5_RIGHT_RCM_GRIPPER_DISTANCE)

        # --- Publishers & Subscribers ---
        self.setup_interfaces('fr5_left')
        self.setup_interfaces('fr5_right')

        # Relative Pose Publishers
        self.pub_left_wrt_right = self.create_publisher(Pose, '/fr5/left_gripper_wrt_right_gripper', 10)
        self.pub_right_wrt_left = self.create_publisher(Pose, '/fr5/right_gripper_wrt_left_gripper', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # --- Main Control Loop ---
        self.last_control_time = self.get_clock().now()
        self.timer = self.create_timer(CMD_DT, self.control_loop)
        self.get_logger().info("Dual FR5 Integrated Controller Started.")

    def setup_kdl(self):
        # Build Chain (Same for both)
        self.chain = kdl.Chain()
        self.chain.addSegment(kdl.Segment("j1", kdl.Joint("j1", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(1.5708, 0, 0), kdl.Vector(0, 0, 0.152))))
        self.chain.addSegment(kdl.Segment("j2", kdl.Joint("j2", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(0, 0, 0), kdl.Vector(-0.425, 0, 0))))
        self.chain.addSegment(kdl.Segment("j3", kdl.Joint("j3", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(0, 0, 0), kdl.Vector(-0.39501, 0, 0))))
        self.chain.addSegment(kdl.Segment("j4", kdl.Joint("j4", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(1.5708, 0, 0), kdl.Vector(0, 0, 0.1021))))
        self.chain.addSegment(kdl.Segment("j5", kdl.Joint("j5", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(-1.5708, 0, 0), kdl.Vector(0, 0, 0.102))))
        self.chain.addSegment(kdl.Segment("j6", kdl.Joint("j6", kdl.Joint.RotZ), kdl.Frame.Identity()))

        # Solvers
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.ik_v_solver = kdl.ChainIkSolverVel_pinv(self.chain)
        self.ik_p_solver = kdl.ChainIkSolverPos_NR(self.chain, self.fk_solver, self.ik_v_solver, 100, 1e-4)

        # Base Transforms (For Relative Pose)
        d = FR5_BASE_DISTANCE
        angle = np.radians(135)
        self.T_LeftBase_RightBase = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(d*np.cos(angle), d*np.sin(angle), 0))

    def connect_robot(self, name, ip):
        try:
            self.get_logger().info(f"Connecting to {name} at {ip}...")
            robot = Robot.RPC(ip)
            self.get_logger().info(f"{name} connected.")
            return robot
        except Exception as e:
            self.get_logger().error(f"{name} Connection Error: {e}")
            return None

    def init_robot_state(self, name, robot, gripper_np, rcm_dist):
        """
        로봇 초기화 및 그리퍼 시퀀스 실행 (Reset -> 5.0 -> 100.0)
        """
        state = {
            'name': name,
            'robot': robot,
            'current_q_deg': [0.0]*6,
            'target_pose_rcm': None,
            'last_pose_time': self.get_clock().now(),
            'cmd_id': 0,
            'last_gripper_val': 100.0,
            'home_pos_deg': [],
            'j6_T_gripper': self.numpy_to_kdl_frame(gripper_np),
            'base_T_rcm': kdl.Frame(),
            'rcm_dist': rcm_dist,
            'gripper_T_j6': self.numpy_to_kdl_frame(gripper_np).Inverse()
        }
        
        if robot.robot_state_pkg:
            state['current_q_deg'] = list(robot.robot_state_pkg.jt_cur_pos)
            state['home_pos_deg'] = list(state['current_q_deg'])
            self.get_logger().info(f"{name} Home Position Saved: {state['home_pos_deg']}")
        
        robot.ServoMoveStart()
        
        # Gripper Reset & Test Sequence
        self.get_logger().info(f"{name} resetting gripper")
        robot.ActGripper(1,0)
        time.sleep(1.0)
        
        self.get_logger().info(f"{name} Activating gripper")
        robot.ActGripper(1,1)
        time.sleep(1.0)

        self.get_logger().info(f"{name} Testing gripper sequence (5 -> 100)...")
        
        # Move to 5.0
        robot.ServoMoveEnd()
        robot.MoveGripper(1, 5.0, 100, 50, 3000, 0, 0, 0, 0, 0)
        time.sleep(2.0)
        
        if robot.robot_state_pkg:
            state['current_q_deg'] = list(robot.robot_state_pkg.jt_cur_pos)
        robot.ServoMoveStart()

        # Move to 100.0
        robot.ServoMoveEnd()
        robot.MoveGripper(1, 100.0, 100, 50, 3000, 0, 0, 0, 0, 0)
        time.sleep(2.0)
        
        if robot.robot_state_pkg:
            state['current_q_deg'] = list(robot.robot_state_pkg.jt_cur_pos)
        robot.ServoMoveStart()
        
        state['last_gripper_val'] = 100.0
        self.get_logger().info(f"{name} Gripper test complete! Good to go.")

        # RCM Frame Init
        q_kdl = self.deg_list_to_kdl_jnt(state['current_q_deg'])
        base_T_j6 = kdl.Frame()
        self.fk_solver.JntToCart(q_kdl, base_T_j6)
        base_T_gripper = base_T_j6 * state['j6_T_gripper']
        gripper_T_rcm = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(-rcm_dist, 0, 0))
        state['base_T_rcm'] = base_T_gripper * gripper_T_rcm
        self.get_logger().info(f"{name} RCM Frame Initialized.")
        
        return state

    def setup_interfaces(self, prefix):
        # Subscribers
        self.create_subscription(Pose, f'/{prefix}/desired_pose_wrt_rcm', lambda m, n=prefix: self.pose_cb(m, n), 10)
        self.create_subscription(Float32, f'/{prefix}/desired_gripper_pose', lambda m, n=prefix: self.gripper_cb(m, n), 10)
        
        # Publishers
        setattr(self, f'{prefix}_joint_pub', self.create_publisher(JointState, f'/{prefix}/joint_states', 10))
        setattr(self, f'{prefix}_vel_cmd_pub', self.create_publisher(JointState, f'/{prefix}/joint_velocity_cmds', 10))
        
        # [수정] 토픽 이름 'current_' 제거 -> '/fr5_left/gripper_wrt_rcm'
        setattr(self, f'{prefix}_rcm_pose_pub', self.create_publisher(Pose, f'/{prefix}/gripper_wrt_rcm', 10))
        
        setattr(self, f'{prefix}_pose_pub', self.create_publisher(Pose, f'/{prefix}/ee_wrt_base', 10))
        setattr(self, f'{prefix}_gripper_state_pub', self.create_publisher(Float32, f'/{prefix}/gripper_state', 10))

    # --- Callbacks ---
    def pose_cb(self, msg, name):
        state = self.left_state if name == 'fr5_left' else self.right_state
        state['target_pose_rcm'] = msg
        state['last_pose_time'] = self.get_clock().now()

    def gripper_cb(self, msg, name):
        state = self.left_state if name == 'fr5_left' else self.right_state
        desired = float(msg.data)
        if desired != state['last_gripper_val']:
            state['last_gripper_val'] = desired
            
            state['robot'].ServoMoveEnd()
            self.get_logger().info(f"{name} Moving gripper to {desired}...")
            state['robot'].MoveGripper(1, desired, 100, 50, 3000, 0, 0, 0, 0, 0)
            
            if state['robot'].robot_state_pkg:
                state['current_q_deg'] = list(state['robot'].robot_state_pkg.jt_cur_pos)
            
            state['robot'].ServoMoveStart()

    # --- Helpers ---
    def numpy_to_kdl_frame(self, mat):
        rot = kdl.Rotation(mat[0,0], mat[0,1], mat[0,2], mat[1,0], mat[1,1], mat[1,2], mat[2,0], mat[2,1], mat[2,2])
        vec = kdl.Vector(mat[0,3], mat[1,3], mat[2,3])
        return kdl.Frame(rot, vec)

    def deg_list_to_kdl_jnt(self, deg_list):
        q = kdl.JntArray(6)
        for i, d in enumerate(deg_list): q[i] = math.radians(d)
        return q

    def pose_msg_to_kdl_frame(self, msg):
        vec = kdl.Vector(msg.position.x, msg.position.y, msg.position.z)
        rot = kdl.Rotation.Quaternion(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        return kdl.Frame(rot, vec)

    def kdl_frame_to_pose_msg(self, frame):
        p = Pose()
        p.position.x = frame.p.x(); p.position.y = frame.p.y(); p.position.z = frame.p.z()
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = frame.M.GetQuaternion()
        return p

    # --- Control Logic ---
    def control_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_control_time).nanoseconds / 1e9
        self.last_control_time = now
        if dt > 0.1: dt = CMD_DT

        # 1. Update State & Calculate Commands for Both Arms
        self.process_arm(self.left_state, dt, 'fr5_left')
        self.process_arm(self.right_state, dt, 'fr5_right')

        # 2. Calculate and Publish Relative Poses
        self.publish_relative_poses()

    def process_arm(self, state, dt, prefix):
        # A. Update Current State
        current_q_deg = state['current_q_deg']
        current_q_rad = self.deg_list_to_kdl_jnt(current_q_deg)

        # B. Inverse Kinematics (Position Control)
        target_q_deg = list(current_q_deg) 
        calculated_vel_rad = [0.0] * 6 

        if state['target_pose_rcm'] is not None:
            lag = (self.get_clock().now() - state['last_pose_time']).nanoseconds / 1e9
            if lag < 0.5:
                rcm_T_grip_des = self.pose_msg_to_kdl_frame(state['target_pose_rcm'])
                base_T_grip_des = state['base_T_rcm'] * rcm_T_grip_des
                base_T_j6_des = base_T_grip_des * state['gripper_T_j6']
                
                q_out = kdl.JntArray(6)
                ret = self.ik_p_solver.CartToJnt(current_q_rad, base_T_j6_des, q_out)
                
                if ret >= 0:
                    target_q_deg = [math.degrees(q_out[i]) for i in range(6)]
                
        # C. Smooth & Clamp
        max_delta = math.degrees(self.max_vel) * dt
        final_cmd_deg = []
        for i in range(6):
            diff = target_q_deg[i] - current_q_deg[i]
            clamped_diff = max(min(diff, max_delta), -max_delta)
            final_cmd_deg.append(current_q_deg[i] + clamped_diff)
            calculated_vel_rad[i] = math.radians(clamped_diff) / dt

        # apply clipping on j6 (-165 ~ 165)
        final_cmd_deg[5] = max(min(final_cmd_deg[5], J6_LIMIT_DEG), -J6_LIMIT_DEG)
        
        state['current_q_deg'] = final_cmd_deg

        # D. Send ServoJ Command
        state['cmd_id'] += 1
        state['robot'].ServoJ(
            joint_pos=final_cmd_deg,
            axisPos=[0.0]*4, acc=0.0, vel=0.0, 
            cmdT=dt, filterT=0.0, gain=0.0, 
            id=state['cmd_id']
        )

        # E. Publish Topics
        now_msg = self.get_clock().now().to_msg()
        
        js_msg = JointState()
        js_msg.header.stamp = now_msg
        js_msg.name = [f'{prefix}_joint{i+1}' for i in range(6)]
        js_msg.position = [math.radians(d) for d in final_cmd_deg]
        getattr(self, f'{prefix}_joint_pub').publish(js_msg)

        vel_msg = JointState()
        vel_msg.header.stamp = now_msg
        vel_msg.velocity = calculated_vel_rad
        getattr(self, f'{prefix}_vel_cmd_pub').publish(vel_msg)

        # Gripper Pose wrt RCM
        base_T_j6 = kdl.Frame()
        self.fk_solver.JntToCart(self.deg_list_to_kdl_jnt(final_cmd_deg), base_T_j6)
        base_T_grip = base_T_j6 * state['j6_T_gripper']
        rcm_T_grip = state['base_T_rcm'].Inverse() * base_T_grip
        
        # [수정] 이전에는 current_gripper_wrt_rcm 이었으나 gripper_wrt_rcm 으로 수정됨
        getattr(self, f'{prefix}_rcm_pose_pub').publish(self.kdl_frame_to_pose_msg(rcm_T_grip))
        
        p_msg = self.kdl_frame_to_pose_msg(base_T_grip)
        getattr(self, f'{prefix}_pose_pub').publish(p_msg)

        g_msg = Float32()
        g_msg.data = state['last_gripper_val']
        getattr(self, f'{prefix}_gripper_state_pub').publish(g_msg)

    def publish_relative_poses(self):
        q_l = self.deg_list_to_kdl_jnt(self.left_state['current_q_deg'])
        q_r = self.deg_list_to_kdl_jnt(self.right_state['current_q_deg'])
        
        fk_l, fk_r = kdl.Frame(), kdl.Frame()
        self.fk_solver.JntToCart(q_l, fk_l)
        self.fk_solver.JntToCart(q_r, fk_r)
        
        T_BaseL_GripL = fk_l * self.left_state['j6_T_gripper']
        T_BaseR_GripR = fk_r * self.right_state['j6_T_gripper']
        T_BaseL_GripR = self.T_LeftBase_RightBase * T_BaseR_GripR
        T_GripR_GripL = T_BaseL_GripR.Inverse() * T_BaseL_GripL
        
        self.pub_left_wrt_right.publish(self.kdl_frame_to_pose_msg(T_GripR_GripL))
        self.pub_right_wrt_left.publish(self.kdl_frame_to_pose_msg(T_GripR_GripL.Inverse()))

    def perform_homing(self):
        self.get_logger().info("Stopping Servo Mode & Homing...")
        
        def home_robot(state):
            try:
                state['robot'].ServoMoveEnd()
                time.sleep(0.5)
                state['robot'].MoveJ(joint_pos=state['home_pos_deg'], tool=0, user=0, vel=15.0, blendT=-1.0)
            except: pass

        t1 = threading.Thread(target=home_robot, args=(self.left_state,))
        t2 = threading.Thread(target=home_robot, args=(self.right_state,))
        t1.start(); t2.start()
        t1.join(); t2.join()
        self.get_logger().info("Homing Complete.")

def main(args=None):
    rclpy.init(args=args)
    node = DualFR5IntegratedController()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    def signal_handler(sig, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.perform_homing()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        node.left_robot.ServoMoveEnd()
        node.right_robot.ServoMoveEnd()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()