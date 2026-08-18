"""
RCM Position Controller for two FR5s (4D State Version: [x, y, z, roll] w.r.t Space Frame)
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Float64MultiArray
import PyKDL as kdl
import numpy as np
import math

# === Configuration ===
CONTROL_RATE_HZ = 100.0  
MAX_LINEAR_VEL = 0.5     
MAX_ANGULAR_VEL = 1.0

# PID Gains (4D: x, y, z, roll)
PID_KP_POS = 4.0   
PID_KP_ROT = 2.0   
PID_KI_POS = 0.0005
PID_KI_ROT = 0.0005
MAX_I_TERM = 0.05

RCM_GAIN_P = 1.0   

# === [OSCILLATION FIX SETTINGS] ===
DEADBAND_POS = 0.0002  # 0.2 mm
DEADBAND_ROT = 0.001   # approx 0.05 deg
RCM_TOLERANCE = 0.001  # 1.0 mm
JOINT_VEL_THRESHOLD = 0.001 # rad/s
# ==================================

# Transformation: J6 -> Gripper
FR5_LEFT_GRIPPER_WRT_J6_TRANS = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.64],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)
FR5_LEFT_RCM_GRIPPER_DISTANCE = 0.190 

FR5_RIGHT_GRIPPER_WRT_J6_TRANS = np.array([
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.64],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)
FR5_RIGHT_RCM_GRIPPER_DISTANCE = 0.192  

FR5_BASE_DISTANCE = 0.61 


def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

def wrap_angle(angle):
    """ 각도를 -pi ~ pi 사이로 정규화 (최단 거리 회전용) """
    return (angle + np.pi) % (2 * np.pi) - np.pi

class PIDController4D:
    """ 4D State (x, y, z, roll) 용 PID 제어기 """
    def __init__(self):
        self.integral = np.zeros(4)

    def compute(self, error_vector, dt):
        if dt <= 1e-6: return np.zeros(4)

        pos_err_norm = np.linalg.norm(error_vector[:3])
        rot_err_norm = abs(error_vector[3])

        # Deadband
        if pos_err_norm < DEADBAND_POS and rot_err_norm < DEADBAND_ROT:
            return np.zeros(4)

        # P-Term
        p_term = np.zeros(4)
        p_term[:3] = error_vector[:3] * PID_KP_POS
        p_term[3]  = error_vector[3] * PID_KP_ROT

        # I-Term
        self.integral[:3] += error_vector[:3] * dt
        self.integral[3]  += error_vector[3] * dt
        self.integral = np.clip(self.integral, -MAX_I_TERM, MAX_I_TERM)
        
        i_term = np.zeros(4)
        i_term[:3] = self.integral[:3] * PID_KI_POS
        i_term[3]  = self.integral[3] * PID_KI_ROT

        return p_term + i_term


class RCMTwoFR5ActionController(Node):
    def __init__(self):
        super().__init__('RCM_two_FR5_action_controller_4D')
        self.chain = self.build_fr5_chain()
        
        self.left_grip_trans = self.numpy_to_kdl_frame(FR5_LEFT_GRIPPER_WRT_J6_TRANS)
        self.right_grip_trans = self.numpy_to_kdl_frame(FR5_RIGHT_GRIPPER_WRT_J6_TRANS)
        
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)

        # --- Space Frame Definition ---
        self.R_space_base = kdl.Rotation.RotZ(5.0 * math.pi / 4.0)
        self.R_base_space = self.R_space_base.Inverse()

        # PID Controllers
        self.pid_left = PIDController4D()
        self.pid_right = PIDController4D()

        # --- State Variables ---
        self.left_q = None
        self.right_q = None
        
        # Target 4D States: np.array([x, y, z, roll])
        self.left_target_state = None
        self.right_target_state = None
        
        # RCM Positions (kdl.Vector in Base Frame)
        self.left_rcm_pos_base = None 
        self.right_rcm_pos_base = None

        self.last_left_time = None
        self.last_right_time = None
        self.last_control_time = self.get_clock().now()

        # --- Subscribers & Publishers ---
        self.create_subscription(JointState, '/fr5_left/joint_states', self.left_joint_cb, 10)
        self.create_subscription(JointState, '/fr5_right/joint_states', self.right_joint_cb, 10)
        
        # Action Input: Twist (dx, dy, dz, droll in NEW Space Frame)
        self.create_subscription(Twist, '/fr5_left/action_delta', self.left_action_cb, 10)
        self.create_subscription(Twist, '/fr5_right/action_delta', self.right_action_cb, 10)

        self.pub_left_vel = self.create_publisher(JointState, '/fr5_left/joint_velocity_cmds', 10)
        self.pub_right_vel = self.create_publisher(JointState, '/fr5_right/joint_velocity_cmds', 10)

        # Relative Pose Pub (Optional)
        self.pub_left_wrt_right = self.create_publisher(Pose, '/fr5/left_gripper_wrt_right_gripper', 10)
        self.pub_right_wrt_left = self.create_publisher(Pose, '/fr5/right_gripper_wrt_left_gripper', 10)

        # Base Transform
        d = FR5_BASE_DISTANCE
        angle = np.radians(135)
        self.T_LeftBase_RightBase = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(d*np.cos(angle), d*np.sin(angle), 0))

        # Timer
        self.timer = self.create_timer(1.0 / CONTROL_RATE_HZ, self.control_loop)
        self.get_logger().info(f"4D Space Frame Action Controller Started at {CONTROL_RATE_HZ}Hz")

    # --- Setup Helpers ---
    def numpy_to_kdl_frame(self, mat):
        rot = kdl.Rotation(mat[0,0], mat[0,1], mat[0,2], mat[1,0], mat[1,1], mat[1,2], mat[2,0], mat[2,1], mat[2,2])
        vec = kdl.Vector(mat[0,3], mat[1,3], mat[2,3])
        return kdl.Frame(rot, vec)

    def kdl_frame_to_pose_msg(self, frame):
        p = Pose()
        p.position.x = frame.p.x(); p.position.y = frame.p.y(); p.position.z = frame.p.z()
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = frame.M.GetQuaternion()
        return p

    def build_fr5_chain(self):
        chain = kdl.Chain()
        chain.addSegment(kdl.Segment("j1", kdl.Joint("j1", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(1.5708, 0, 0), kdl.Vector(0, 0, 0.152))))
        chain.addSegment(kdl.Segment("j2", kdl.Joint("j2", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(0, 0, 0), kdl.Vector(-0.425, 0, 0))))
        chain.addSegment(kdl.Segment("j3", kdl.Joint("j3", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(0, 0, 0), kdl.Vector(-0.39501, 0, 0))))
        chain.addSegment(kdl.Segment("j4", kdl.Joint("j4", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(1.5708, 0, 0), kdl.Vector(0, 0, 0.1021))))
        chain.addSegment(kdl.Segment("j5", kdl.Joint("j5", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(-1.5708, 0, 0), kdl.Vector(0, 0, 0.102))))
        chain.addSegment(kdl.Segment("j6", kdl.Joint("j6", kdl.Joint.RotZ), kdl.Frame.Identity()))
        return chain

    def get_kdl_jnt_array(self, msg):
        q = kdl.JntArray(6)
        if len(msg.position) >= 6:
            for i in range(6): q[i] = msg.position[i]
        return q

    # --- 4D Kinematics Calculation ---
    def calculate_initial_rcm_pos(self, q, gripper_transform, rcm_dist):
        frame_j6 = kdl.Frame()
        self.fk_solver.JntToCart(q, frame_j6)
        frame_gripper = frame_j6 * gripper_transform 
        rcm_point = frame_gripper * kdl.Vector(-rcm_dist, 0, 0)
        return rcm_point # kdl.Vector 반환

    def calculate_4d_state(self, q, gripper_transform, rcm_pos_base):
        frame_j6 = kdl.Frame()
        self.fk_solver.JntToCart(q, frame_j6)
        frame_gripper_base = frame_j6 * gripper_transform
        
        vec_rcm_to_g_base = frame_gripper_base.p - rcm_pos_base
        pos_new_rcm = self.R_base_space * vec_rcm_to_g_base
        
        X_g_base = frame_gripper_base.M.UnitX()
        Y_g_base = frame_gripper_base.M.UnitY()
        
        X_g_space = self.R_base_space * X_g_base
        Y_g_space = self.R_base_space * Y_g_base
        
        Z_space = kdl.Vector(0, 0, 1)
        Y_a_unnorm = Z_space * X_g_space 
        
        roll = 0.0
        if Y_a_unnorm.Norm() > 1e-6:
            Y_a = Y_a_unnorm / Y_a_unnorm.Norm()
            cos_roll = kdl.dot(Y_a, Y_g_space)
            sin_roll = kdl.dot(Y_a * Y_g_space, X_g_space)
            roll = math.atan2(sin_roll, cos_roll)

        return np.array([pos_new_rcm.x(), pos_new_rcm.y(), pos_new_rcm.z(), roll], dtype=np.float32)

    # --- Callbacks ---
    def left_joint_cb(self, msg):
        self.left_q = self.get_kdl_jnt_array(msg)
        self.last_left_time = self.get_clock().now()
        
        if self.left_rcm_pos_base is None:
            self.left_rcm_pos_base = self.calculate_initial_rcm_pos(self.left_q, self.left_grip_trans, FR5_LEFT_RCM_GRIPPER_DISTANCE)
            self.left_target_state = self.calculate_4d_state(self.left_q, self.left_grip_trans, self.left_rcm_pos_base)
            self.get_logger().info("Left RCM & 4D Target Initialized")
            
        self.publish_relative_poses()

    def right_joint_cb(self, msg):
        self.right_q = self.get_kdl_jnt_array(msg)
        self.last_right_time = self.get_clock().now()
        
        if self.right_rcm_pos_base is None:
            self.right_rcm_pos_base = self.calculate_initial_rcm_pos(self.right_q, self.right_grip_trans, FR5_RIGHT_RCM_GRIPPER_DISTANCE)
            self.right_target_state = self.calculate_4d_state(self.right_q, self.right_grip_trans, self.right_rcm_pos_base)
            self.get_logger().info("Right RCM & 4D Target Initialized")
            
        self.publish_relative_poses()

    # --- Action Integration ---
    def left_action_cb(self, msg):
        """ Update Left Target 4D State based on Delta """
        if self.left_target_state is not None:
            self.left_target_state[0] += msg.linear.x
            self.left_target_state[1] += msg.linear.y
            self.left_target_state[2] += msg.linear.z
            self.left_target_state[3] += msg.angular.x
            self.left_target_state[3] = wrap_angle(self.left_target_state[3])

    def right_action_cb(self, msg):
        """ Update Right Target 4D State based on Delta """
        if self.right_target_state is not None:
            self.right_target_state[0] += msg.linear.x
            self.right_target_state[1] += msg.linear.y
            self.right_target_state[2] += msg.linear.z
            self.right_target_state[3] += msg.angular.x
            self.right_target_state[3] = wrap_angle(self.right_target_state[3])

    # --- Main Control Loop (100Hz) ---
    def control_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_control_time).nanoseconds / 1e9
        self.last_control_time = now
        if dt > 0.1: dt = 0.01

        # --- LEFT ARM ---
        if self.check_ready(self.left_q, self.left_target_state, self.last_left_time):
            # 1. Get Current 4D State
            curr_state = self.calculate_4d_state(self.left_q, self.left_grip_trans, self.left_rcm_pos_base)
            
            # 2. Calculate Error (Target - Current)
            err = self.left_target_state - curr_state
            err[3] = wrap_angle(err[3]) # Shortest rotation path
            
            # 3. PID Control -> Desired Velocity in Space Frame
            pid_vel = self.pid_left.compute(err, dt)
            
            # 4. Solve RCM IK
            q_dot = self.solve_rcm_velocity(self.left_q, pid_vel, self.left_grip_trans, self.left_rcm_pos_base)
            
            if np.max(np.abs(q_dot)) < JOINT_VEL_THRESHOLD:
                q_dot = np.zeros(6)
            self.publish_joint_vel(q_dot, self.pub_left_vel)

        # --- RIGHT ARM ---
        if self.check_ready(self.right_q, self.right_target_state, self.last_right_time):
            curr_state = self.calculate_4d_state(self.right_q, self.right_grip_trans, self.right_rcm_pos_base)
            err = self.right_target_state - curr_state
            err[3] = wrap_angle(err[3])
            
            pid_vel = self.pid_right.compute(err, dt)
            q_dot = self.solve_rcm_velocity(self.right_q, pid_vel, self.right_grip_trans, self.right_rcm_pos_base)
            
            if np.max(np.abs(q_dot)) < JOINT_VEL_THRESHOLD:
                q_dot = np.zeros(6)
            self.publish_joint_vel(q_dot, self.pub_right_vel)

    # --- Core 4D RCM Logic ---
    def solve_rcm_velocity(self, q, pid_4d_vel, gripper_transform, rcm_pos_base):
        """
        pid_4d_vel: [vx_space, vy_space, vz_space, w_roll] (Desired velocities from PID)
        """
        # 1. Velocity Clamping
        v_space_cmd = pid_4d_vel[:3]
        v_norm = np.linalg.norm(v_space_cmd)
        if v_norm > MAX_LINEAR_VEL:
            v_space_cmd = (v_space_cmd / v_norm) * MAX_LINEAR_VEL
            
        w_roll_cmd = clamp(pid_4d_vel[3], -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)

        # 2. FK & Jacobian
        frame_j6 = kdl.Frame()
        self.fk_solver.JntToCart(q, frame_j6)
        frame_gripper = frame_j6 * gripper_transform
        
        jac_j6 = kdl.Jacobian(6)
        self.jac_solver.JntToJac(q, jac_j6)
        offset_j6_grip = frame_j6.M * gripper_transform.p 
        jac_j6.changeRefPoint(offset_j6_grip)
        
        J_base = np.zeros((6, 6))
        for i in range(6):
            for j in range(6): J_base[i, j] = jac_j6[i, j]

        # 3. Kinematic Properties
        R_base_grip = frame_gripper.M
        X_g_base = R_base_grip.UnitX() # Insertion axis
        
        r_rcm_to_grip = frame_gripper.p - rcm_pos_base
        dist_sq = kdl.dot(r_rcm_to_grip, r_rcm_to_grip)
        if dist_sq < 1e-6: dist_sq = 1e-6

        # --- Transform Space Command to Base Frame ---
        v_base_cmd = self.R_space_base * kdl.Vector(v_space_cmd[0], v_space_cmd[1], v_space_cmd[2])

        # --- Decompose Velocity for RCM ---
        # 선속도(v_base_cmd)를 툴 축 방향(Insertion)과 직교 방향(Lateral)으로 분해
        v_ins_base = X_g_base * kdl.dot(v_base_cmd, X_g_base) 
        v_lat_base = v_base_cmd - v_ins_base
        
        # 샤프트를 직교 방향으로 밀어내기 위해 필요한 피벗(Pivot) 회전 속도 계산
        # w = (r x v) / |r|^2
        w_pivot_induced = (r_rcm_to_grip * v_lat_base) / dist_sq
        
        # 툴 자체 롤(Roll) 속도를 Base Frame 벡터로 변환
        w_roll_base = X_g_base * w_roll_cmd
        
        # 최종 로봇 명령 조합
        # v_total_base: 선속도는 PID가 원했던 v_base_cmd를 그대로 반영
        # w_total_base: 롤 + RCM유지용 피벗 회전
        w_total_base = w_roll_base + w_pivot_induced

        # --- Drift Correction (RCM 오차 보정) ---
        vec_grip_rcm = rcm_pos_base - frame_gripper.p
        projection_length = kdl.dot(vec_grip_rcm, X_g_base) 
        closest_point = frame_gripper.p + X_g_base * projection_length
        error_vec = rcm_pos_base - closest_point
        
        if error_vec.Norm() < RCM_TOLERANCE:
            v_correction_base = kdl.Vector(0,0,0)
        else:
            v_correction_base = error_vec * RCM_GAIN_P

        v_total_base = v_base_cmd + v_correction_base
        
        V_cmd_solver = np.array([
            v_total_base.x(), v_total_base.y(), v_total_base.z(), 
            w_total_base.x(), w_total_base.y(), w_total_base.z()
        ])
        
        # 4. IK Solve (Damped Least Squares)
        damping = 0.02 
        A = np.dot(J_base, J_base.T) + (damping**2) * np.eye(6)
        x = np.linalg.solve(A, V_cmd_solver)
        return np.dot(J_base.T, x)

    # --- Utils ---
    def check_ready(self, q, target_state, last_time):
        if q is None or target_state is None or last_time is None: return False
        diff = (self.get_clock().now() - last_time).nanoseconds / 1e9
        if diff > 0.2: return False
        return True

    def get_current_pose_wrt_base(self, q, grip_trans):
        fk = kdl.Frame()
        self.fk_solver.JntToCart(q, fk)
        return fk * grip_trans

    def publish_joint_vel(self, q_dot, publisher):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.velocity = q_dot.flatten().tolist()
        publisher.publish(msg)

    def publish_relative_poses(self): 
        if self.left_q is None or self.right_q is None: return
        fk_l = self.get_current_pose_wrt_base(self.left_q, self.left_grip_trans)
        fk_r = self.get_current_pose_wrt_base(self.right_q, self.right_grip_trans)
        
        T_BaseL_GripR = self.T_LeftBase_RightBase * fk_r
        T_GripR_GripL = T_BaseL_GripR.Inverse() * fk_l
        
        self.pub_left_wrt_right.publish(self.kdl_frame_to_pose_msg(T_GripR_GripL))
        self.pub_right_wrt_left.publish(self.kdl_frame_to_pose_msg(T_GripR_GripL.Inverse()))

def main(args=None):
    rclpy.init(args=args)
    node = RCMTwoFR5ActionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()