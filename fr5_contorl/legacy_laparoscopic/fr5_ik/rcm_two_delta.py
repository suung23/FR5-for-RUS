"""
RCM Position Controller for two FR5s (Oscillation Fixed Version + Full Original Features).
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Pose
import PyKDL as kdl
import numpy as np

# === Configuration ===
CONTROL_RATE_HZ = 100.0  # High frequency control loop
MAX_LINEAR_VEL = 0.5     # Safety limit
MAX_ANGULAR_VEL = 1.0

# PID Gains (Position/Orientation Error -> Velocity)
PID_KP_POS = 4.0   # Position gain
PID_KP_ROT = 2.0   # Rotation gain (Roll)
PID_KI_POS = 0.0005
PID_KI_ROT = 0.0005
MAX_I_TERM = 0.05

RCM_GAIN_P = 1.0   # Strong gain to keep shaft at RCM

# === [OSCILLATION FIX SETTINGS] ===
# 1. PID Deadband: 오차가 이 값보다 작으면 제어 입력을 0으로 함
DEADBAND_POS = 0.0002  # 0.5 mm
DEADBAND_ROT = 0.001   # approx 0.1 deg

# 2. RCM Tolerance: 샤프트가 RCM 포인트에서 이 거리 이내면 보정 안 함
RCM_TOLERANCE = 0.001  # 1.0 mm

# 3. Velocity Cutoff: 계산된 Joint 속도가 이보다 작으면 0으로 전송
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

class PIDController6D:
    def __init__(self):
        self.integral = np.zeros(6)

    def compute(self, error_vector, dt):
        # Only P and I are usually enough for this tracking
        if dt <= 1e-6: return np.zeros(6)

        # === [FIX 1] PID Deadband ===
        # 위치 및 회전 오차의 크기(Norm) 계산
        pos_err_norm = np.linalg.norm(error_vector[:3])
        rot_err_norm = np.linalg.norm(error_vector[3:])

        # 오차가 매우 작으면 0 반환 (진동의 주 원인 차단)
        if pos_err_norm < DEADBAND_POS and rot_err_norm < DEADBAND_ROT:
            return np.zeros(6)
        # ============================

        # P-Term
        p_term = np.zeros(6)
        p_term[:3] = error_vector[:3] * PID_KP_POS
        p_term[3:] = error_vector[3:] * PID_KP_ROT

        # I-Term
        self.integral[:3] += error_vector[:3] * dt
        self.integral[3:] += error_vector[3:] * dt
        self.integral = np.clip(self.integral, -MAX_I_TERM, MAX_I_TERM)
        
        i_term = np.zeros(6)
        i_term[:3] = self.integral[:3] * PID_KI_POS
        i_term[3:] = self.integral[3:] * PID_KI_ROT

        return p_term + i_term


class RCMTwoFR5ActionController(Node):
    def __init__(self):
        super().__init__('RCM_two_FR5_action_controller')
        self.chain = self.build_fr5_chain()
        
        self.left_grip_trans = self.numpy_to_kdl_frame(FR5_LEFT_GRIPPER_WRT_J6_TRANS)
        self.right_grip_trans = self.numpy_to_kdl_frame(FR5_RIGHT_GRIPPER_WRT_J6_TRANS)
        
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)

        # PID Controllers
        self.pid_left = PIDController6D()
        self.pid_right = PIDController6D()

        # --- State Variables ---
        self.left_q = None
        self.right_q = None
        
        # Target Poses (KDL Frame) - Updated by Action Delta
        self.left_target_frame = None
        self.right_target_frame = None
        
        self.left_rcm_base = None 
        self.right_rcm_base = None

        self.last_left_time = None
        self.last_right_time = None
        self.last_control_time = self.get_clock().now()

        # --- Subscribers & Publishers ---
        self.create_subscription(JointState, '/fr5_left/joint_states', self.left_joint_cb, 10)
        self.create_subscription(JointState, '/fr5_right/joint_states', self.right_joint_cb, 10)
        
        # Action Input: Twist (dx, dy, dz, droll)
        self.create_subscription(Twist, '/fr5_left/action_delta', self.left_action_cb, 10)
        self.create_subscription(Twist, '/fr5_right/action_delta', self.right_action_cb, 10)

        self.pub_left_vel = self.create_publisher(JointState, '/fr5_left/joint_velocity_cmds', 10)
        self.pub_left_rcm_pose = self.create_publisher(Pose, '/fr5_left/gripper_wrt_rcm', 10)
        self.pub_right_vel = self.create_publisher(JointState, '/fr5_right/joint_velocity_cmds', 10)
        self.pub_right_rcm_pose = self.create_publisher(Pose, '/fr5_right/gripper_wrt_rcm', 10)

        # Relative Pose Pub (Optional) - [KEPT AS ORIGINAL]
        self.pub_left_wrt_right = self.create_publisher(Pose, '/fr5/left_gripper_wrt_right_gripper', 10)
        self.pub_right_wrt_left = self.create_publisher(Pose, '/fr5/right_gripper_wrt_left_gripper', 10)

        # Base Transform - [KEPT AS ORIGINAL]
        d = FR5_BASE_DISTANCE
        angle = np.radians(135)
        self.T_LeftBase_RightBase = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(d*np.cos(angle), d*np.sin(angle), 0))

        # Timer
        self.timer = self.create_timer(1.0 / CONTROL_RATE_HZ, self.control_loop)
        self.get_logger().info(f"RCM Action Controller (Oscillation Fixed) Started at {CONTROL_RATE_HZ}Hz")

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

    # --- Callbacks ---
    def left_joint_cb(self, msg):
        self.left_q = self.get_kdl_jnt_array(msg)
        self.last_left_time = self.get_clock().now()
        
        # Initialize RCM and Target on first run
        if self.left_rcm_base is None:
            self.left_rcm_base = self.calculate_initial_rcm(self.left_q, self.left_grip_trans, FR5_LEFT_RCM_GRIPPER_DISTANCE)
            # Init target as current pose
            self.left_target_frame = self.get_current_pose_wrt_base(self.left_q, self.left_grip_trans)
            self.get_logger().info("Left RCM & Target Initialized")
            
        self.publish_gripper_wrt_rcm(self.left_q, self.left_grip_trans, self.left_rcm_base, self.pub_left_rcm_pose)
        self.publish_relative_poses() # [KEPT AS ORIGINAL]

    def right_joint_cb(self, msg):
        self.right_q = self.get_kdl_jnt_array(msg)
        self.last_right_time = self.get_clock().now()
        
        if self.right_rcm_base is None:
            self.right_rcm_base = self.calculate_initial_rcm(self.right_q, self.right_grip_trans, FR5_RIGHT_RCM_GRIPPER_DISTANCE)
            self.right_target_frame = self.get_current_pose_wrt_base(self.right_q, self.right_grip_trans)
            self.get_logger().info("Right RCM & Target Initialized")
            
        self.publish_gripper_wrt_rcm(self.right_q, self.right_grip_trans, self.right_rcm_base, self.pub_right_rcm_pose)
        self.publish_relative_poses() # [KEPT AS ORIGINAL]

    # --- Action Integration (Twist Delta -> Target Frame) ---
    def integrate_delta(self, current_target_base, twist_msg):
        """
        Applies local delta (dx, dy, dz, droll) to the global target frame.
        current_target_base: KDL Frame (Global)
        twist_msg: geometry_msgs/Twist (Local Delta)
        """
        if current_target_base is None: return None

        # 1. Local Translation Delta (Body Frame)
        d_pos_local = kdl.Vector(twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z)
        d_pos_global = current_target_base.M * d_pos_local # Rotate to global
        
        # 2. Local Rotation Delta (Body Frame - Roll only)
        # We only care about Roll around the X-axis of the gripper
        d_rot_local = kdl.Rotation.RPY(twist_msg.angular.x, 0, 0)
        
        # Correct Rotation Composition: R_new = R_old * R_delta
        new_rot = current_target_base.M * d_rot_local

        # 3. New Frame Construction
        new_pos = current_target_base.p + d_pos_global
        return kdl.Frame(new_rot, new_pos)

    def left_action_cb(self, msg):
        """ Update Left Target Pose based on Delta """
        if self.left_target_frame is not None:
            self.left_target_frame = self.integrate_delta(self.left_target_frame, msg)

    def right_action_cb(self, msg):
        """ Update Right Target Pose based on Delta """
        if self.right_target_frame is not None:
            self.right_target_frame = self.integrate_delta(self.right_target_frame, msg)

    # --- Main Control Loop (100Hz) ---
    def control_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_control_time).nanoseconds / 1e9
        self.last_control_time = now
        
        if dt > 0.1: dt = 0.01

        # --- LEFT ARM ---
        if self.check_ready(self.left_q, self.left_target_frame, self.last_left_time):
            # 1. Calculate Error (Target - Current) in Body Frame
            F_curr = self.get_current_pose_wrt_base(self.left_q, self.left_grip_trans)
            error_twist = kdl.diff(F_curr, self.left_target_frame) # Global Twist
            error_body = F_curr.M.Inverse() * error_twist # Local Twist
            
            # 2. PID Control -> Desired Velocity (Local Frame)
            err_vec = np.array([
                error_body.vel.x(), error_body.vel.y(), error_body.vel.z(),
                error_body.rot.x(), error_body.rot.y(), error_body.rot.z()
            ])
            pid_vel = self.pid_left.compute(err_vec, dt)
            
            # 3. Solve RCM IK
            q_dot = self.solve_rcm_velocity(self.left_q, pid_vel, self.left_grip_trans, self.left_rcm_base.p)
            
            # === [FIX 3] Velocity Cutoff ===
            if np.max(np.abs(q_dot)) < JOINT_VEL_THRESHOLD:
                q_dot = np.zeros(6)
            # ===============================
            
            self.publish_joint_vel(q_dot, self.pub_left_vel)

        # --- RIGHT ARM ---
        if self.check_ready(self.right_q, self.right_target_frame, self.last_right_time):
            F_curr = self.get_current_pose_wrt_base(self.right_q, self.right_grip_trans)
            error_twist = kdl.diff(F_curr, self.right_target_frame)
            error_body = F_curr.M.Inverse() * error_twist
            
            err_vec = np.array([
                error_body.vel.x(), error_body.vel.y(), error_body.vel.z(),
                error_body.rot.x(), error_body.rot.y(), error_body.rot.z()
            ])
            pid_vel = self.pid_right.compute(err_vec, dt)
            
            q_dot = self.solve_rcm_velocity(self.right_q, pid_vel, self.right_grip_trans, self.right_rcm_base.p)
            
            # === [FIX 3] Velocity Cutoff ===
            if np.max(np.abs(q_dot)) < JOINT_VEL_THRESHOLD:
                q_dot = np.zeros(6)
            # ===============================

            self.publish_joint_vel(q_dot, self.pub_right_vel)

    # --- Core Logic ---
    def solve_rcm_velocity(self, q, velocity_cmd, gripper_transform, rcm_point_base):
        """
        velocity_cmd: [vx, vy, vz, wx, wy, wz] (Desired Velocity in BODY Frame output by PID)
        """
        # 1. FK & Jacobian
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

        # 2. Vector Basis
        R_base_grip = frame_gripper.M
        r_rcm_to_grip = frame_gripper.p - rcm_point_base
        dist_sq = kdl.dot(r_rcm_to_grip, r_rcm_to_grip)
        if dist_sq < 1e-6: dist_sq = 1e-6

        # --- DECOUPLING (PID Velocity -> RCM Compatible Velocity) ---
        
        # A. Insertion (Driven by PID X output)
        v_ins_val = clamp(velocity_cmd[0], -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
        v_ins_base = R_base_grip * kdl.Vector(v_ins_val, 0, 0)
        
        # B. Roll (Driven by PID Roll output)
        w_roll_val = clamp(velocity_cmd[3], -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
        w_roll_base = R_base_grip * kdl.Vector(w_roll_val, 0, 0)

        # C. Pivot (Driven by PID Y/Z output)
        # PID says: "Move tip Y by dy, Z by dz"
        dy = clamp(velocity_cmd[1], -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
        dz = clamp(velocity_cmd[2], -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
        v_lateral_base = R_base_grip * kdl.Vector(0, dy, dz)
        
        # Convert this lateral velocity into Pivot Rotation (w)
        w_pivot_induced = (r_rcm_to_grip * v_lateral_base) / dist_sq
        
        # Re-convert to linear to ensure orthogonality
        v_pivot_clean = w_pivot_induced * r_rcm_to_grip

        # D. RCM Correction
        axis_shaft_base = R_base_grip.UnitX()
        vec_grip_rcm = rcm_point_base - frame_gripper.p
        projection_length = kdl.dot(vec_grip_rcm, axis_shaft_base) 
        closest_point = frame_gripper.p + axis_shaft_base * projection_length
        error_vec = rcm_point_base - closest_point
        
        # === [FIX 2] RCM Tolerance ===
        # 만약 RCM 오차가 허용범위(Tolerance) 내라면 보정력을 0으로 설정
        if error_vec.Norm() < RCM_TOLERANCE:
            v_correction_base = kdl.Vector(0,0,0)
        else:
            v_correction_base = error_vec * RCM_GAIN_P
        # =============================

        # 3. Synthesis
        v_total_base = v_ins_base + v_pivot_clean + v_correction_base
        w_total_base = w_roll_base + w_pivot_induced
        
        V_cmd_solver = np.array([
            v_total_base.x(), v_total_base.y(), v_total_base.z(), 
            w_total_base.x(), w_total_base.y(), w_total_base.z()
        ])
        
        # 4. IK
        damping = 0.02 
        A = np.dot(J_base, J_base.T) + (damping**2) * np.eye(6)
        x = np.linalg.solve(A, V_cmd_solver)
        return np.dot(J_base.T, x)

    # --- Utils ---
    def check_ready(self, q, target_frame, last_time):
        if q is None or target_frame is None or last_time is None: return False
        diff = (self.get_clock().now() - last_time).nanoseconds / 1e9
        if diff > 0.2: return False
        return True

    def get_current_pose_wrt_base(self, q, grip_trans):
        fk = kdl.Frame()
        self.fk_solver.JntToCart(q, fk)
        return fk * grip_trans

    def get_current_pose_wrt_rcm(self, q, grip_trans, rcm_base):
        fk = self.get_current_pose_wrt_base(q, grip_trans)
        return rcm_base.Inverse() * fk

    def calculate_initial_rcm(self, q, gripper_transform, rcm_dist):
        frame_j6 = kdl.Frame()
        self.fk_solver.JntToCart(q, frame_j6)
        frame_gripper = frame_j6 * gripper_transform 
        rcm_point = frame_gripper * kdl.Vector(-rcm_dist, 0, 0)
        return kdl.Frame(frame_gripper.M, rcm_point)

    def publish_joint_vel(self, q_dot, publisher):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.velocity = q_dot.flatten().tolist()
        publisher.publish(msg)

    def publish_gripper_wrt_rcm(self, q, grip_trans, rcm_frame_base, publisher):
        pose = self.kdl_frame_to_pose_msg(self.get_current_pose_wrt_rcm(q, grip_trans, rcm_frame_base))
        publisher.publish(pose)

    def publish_relative_poses(self): # [KEPT AS ORIGINAL]
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