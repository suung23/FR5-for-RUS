"""
RCM Position Controller for two FR5s (Updated with Linear-Angular Blending).
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Pose
import PyKDL as kdl
import numpy as np
import math

### PID Constants (Updated) ###
# Reduced P, Removed D to prevent noise amplification
PID_KP_POS = 2.0  
PID_KI_POS = 0.01 
PID_KD_POS = 0.0  # Set to 0.0 to stop vibration from teleop noise

PID_KP_ROT = 1.0
PID_KI_ROT = 0.01
PID_KD_ROT = 0.0  # Set to 0.0


MAX_I_TERM = 0.05
MAX_LINEAR_VEL = 1.0
MAX_ANGULAR_VEL = 1.0
JOINT_TIMEOUT_SEC = 0.2
CONTROL_RATE_HZ = 100.0 # 10ms period
# --- Blending Weights (Fixed) ---
# Use Position Error to drive the pivoting. 
# Use Orientation Error ONLY for Roll (handled separately).
WEIGHT_LINEAR_ERROR = 1.0   
WEIGHT_ANGULAR_ERROR = 0.0  # Set to 0 to avoid double-counting/fighting

RCM_GAIN_P = 1.0 # Reduced from 3.0 to soften the constraint

# Transformation: J6 -> Gripper
FR5_LEFT_GRIPPER_WRT_J6_TRANS = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.64],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

FR5_LEFT_RCM_GRIPPER_DISTANCE = 0.190 # Meters (Distance along negative X-axis)

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
        self.prev_error = np.zeros(6)
        self.integral = np.zeros(6)
        self.first_run = True

    def compute(self, error_vector, dt):
        if dt <= 0.000001: return np.zeros(6)

        # P
        p_term = np.zeros(6)
        p_term[:3] = error_vector[:3] * PID_KP_POS
        p_term[3:] = error_vector[3:] * PID_KP_ROT

        # I
        self.integral[:3] += error_vector[:3] * dt
        self.integral[3:] += error_vector[3:] * dt
        self.integral = np.clip(self.integral, -MAX_I_TERM, MAX_I_TERM)
        
        i_term = np.zeros(6)
        i_term[:3] = self.integral[:3] * PID_KI_POS
        i_term[3:] = self.integral[3:] * PID_KI_ROT

        # D
        if self.first_run:
            d_term = np.zeros(6)
            self.first_run = False
        else:
            delta_error = error_vector - self.prev_error
            derivative = delta_error / dt
            d_term = np.zeros(6)
            d_term[:3] = derivative[:3] * PID_KD_POS
            d_term[3:] = derivative[3:] * PID_KD_ROT

        self.prev_error = error_vector
        return p_term + i_term + d_term


class RCMTwoFR5HighFreqController(Node):
    def __init__(self):
        super().__init__('RCM_two_FR5_high_freq_controller')
        self.chain = self.build_fr5_chain()
        
        self.left_gripper_frame_kdl = self.numpy_to_kdl_frame(FR5_LEFT_GRIPPER_WRT_J6_TRANS)
        self.right_gripper_frame_kdl = self.numpy_to_kdl_frame(FR5_RIGHT_GRIPPER_WRT_J6_TRANS)
        
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)

        # PID
        self.pid_left = PIDController6D()
        self.pid_right = PIDController6D()
        
        # --- State Variables ---
        self.left_q = None
        self.right_q = None
        
        self.left_des_pose_msg = None  
        self.right_des_pose_msg = None 

        self.last_left_joint_time = None
        self.last_right_joint_time = None
        self.last_control_time = self.get_clock().now()
        
        self.left_rcm_frame_base = None 
        self.right_rcm_frame_base = None

        # --- Publishers & Subscribers ---
        self.create_subscription(JointState, '/fr5_left/joint_states', self.left_joint_cb, 10)
        self.create_subscription(Pose, '/fr5_left/desired_pose_wrt_rcm', self.left_pose_cb, 10)
        self.pub_left_vel = self.create_publisher(JointState, '/fr5_left/joint_velocity_cmds', 10)
        self.pub_left_rcm_pose = self.create_publisher(Pose, '/fr5_left/gripper_wrt_rcm', 10)
        
        self.create_subscription(JointState, '/fr5_right/joint_states', self.right_joint_cb, 10)
        self.create_subscription(Pose, '/fr5_right/desired_pose_wrt_rcm', self.right_pose_cb, 10)
        self.pub_right_vel = self.create_publisher(JointState, '/fr5_right/joint_velocity_cmds', 10)
        self.pub_right_rcm_pose = self.create_publisher(Pose, '/fr5_right/gripper_wrt_rcm', 10)

        self.pub_left_wrt_right = self.create_publisher(Pose, '/fr5/left_gripper_wrt_right_gripper', 10)
        self.pub_right_wrt_left = self.create_publisher(Pose, '/fr5/right_gripper_wrt_left_gripper', 10)

        # Base Transform
        d = FR5_BASE_DISTANCE
        angle = np.radians(135)
        self.T_LeftBase_RightBase = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(d*np.cos(angle), d*np.sin(angle), 0))

        # --- High Frequency Timer ---
        timer_period = 1.0 / CONTROL_RATE_HZ
        self.timer = self.create_timer(timer_period, self.control_loop)
        self.get_logger().info(f"RCM Controller Started at {CONTROL_RATE_HZ}Hz")

    # --- Setup Helpers ---
    def numpy_to_kdl_frame(self, mat):
        rot = kdl.Rotation(mat[0,0], mat[0,1], mat[0,2], mat[1,0], mat[1,1], mat[1,2], mat[2,0], mat[2,1], mat[2,2])
        vec = kdl.Vector(mat[0,3], mat[1,3], mat[2,3])
        return kdl.Frame(rot, vec)

    def pose_msg_to_kdl_frame(self, msg):
        vec = kdl.Vector(msg.position.x, msg.position.y, msg.position.z)
        rot = kdl.Rotation.Quaternion(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
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
        self.last_left_joint_time = self.get_clock().now()
        if self.left_rcm_frame_base is None:
            self.left_rcm_frame_base = self.calculate_initial_rcm(self.left_q, self.left_gripper_frame_kdl, FR5_LEFT_RCM_GRIPPER_DISTANCE)
            self.get_logger().info("Left RCM Initialized")
        self.publish_gripper_wrt_rcm(self.left_q, self.left_gripper_frame_kdl, self.left_rcm_frame_base, self.pub_left_rcm_pose)
        self.publish_relative_poses()

    def right_joint_cb(self, msg):
        self.right_q = self.get_kdl_jnt_array(msg)
        self.last_right_joint_time = self.get_clock().now()
        if self.right_rcm_frame_base is None:
            self.right_rcm_frame_base = self.calculate_initial_rcm(self.right_q, self.right_gripper_frame_kdl, FR5_RIGHT_RCM_GRIPPER_DISTANCE)
            self.get_logger().info("Right RCM Initialized")
        self.publish_gripper_wrt_rcm(self.right_q, self.right_gripper_frame_kdl, self.right_rcm_frame_base, self.pub_right_rcm_pose)
        self.publish_relative_poses()

    def left_pose_cb(self, pose_msg):
        self.left_des_pose_msg = pose_msg

    def right_pose_cb(self, pose_msg):
        self.right_des_pose_msg = pose_msg

    # --- Main Control Loop (100Hz) ---
    def control_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_control_time).nanoseconds / 1e9
        self.last_control_time = now
        
        if dt > 0.1: dt = 0.01

        # --- LEFT ARM ---
        if self.check_ready(self.left_q, self.left_des_pose_msg, self.left_rcm_frame_base, self.last_left_joint_time, "LEFT"):
            F_des = self.pose_msg_to_kdl_frame(self.left_des_pose_msg)
            F_curr = self.get_current_pose_wrt_rcm(self.left_q, self.left_gripper_frame_kdl, self.left_rcm_frame_base)
            
            error_vec = self.compute_error_vector(F_curr, F_des)
            pid_out = self.pid_left.compute(error_vec, dt) # 6D Error PID Output
            
            twist_cmd = self.vector_to_twist_msg(pid_out)
            q_dot = self.solve_rcm_velocity(self.left_q, twist_cmd, self.left_gripper_frame_kdl, self.left_rcm_frame_base.p)
            self.publish_joint_vel(q_dot, self.pub_left_vel)

        # --- RIGHT ARM ---
        if self.check_ready(self.right_q, self.right_des_pose_msg, self.right_rcm_frame_base, self.last_right_joint_time, "RIGHT"):
            F_des = self.pose_msg_to_kdl_frame(self.right_des_pose_msg)
            F_curr = self.get_current_pose_wrt_rcm(self.right_q, self.right_gripper_frame_kdl, self.right_rcm_frame_base)
            
            error_vec = self.compute_error_vector(F_curr, F_des)
            pid_out = self.pid_right.compute(error_vec, dt)
            
            twist_cmd = self.vector_to_twist_msg(pid_out)
            q_dot = self.solve_rcm_velocity(self.right_q, twist_cmd, self.right_gripper_frame_kdl, self.right_rcm_frame_base.p)
            self.publish_joint_vel(q_dot, self.pub_right_vel)

    # --- Helpers ---
    def check_ready(self, q, des_pose, rcm_base, last_joint_time, name):
        if q is None or des_pose is None or rcm_base is None: return False
        if last_joint_time is None: return False
        diff = (self.get_clock().now() - last_joint_time).nanoseconds / 1e9
        if diff > JOINT_TIMEOUT_SEC:
            self.get_logger().warn(f"[{name}] Stale joints! {diff:.3f}s", throttle_duration_sec=1.0)
            return False
        return True

    def get_current_pose_wrt_rcm(self, q, grip_trans, rcm_base):
        fk = kdl.Frame()
        self.fk_solver.JntToCart(q, fk)
        return rcm_base.Inverse() * (fk * grip_trans)

    def compute_error_vector(self, F_current, F_desired):
        twist_rcm = kdl.diff(F_current, F_desired)
        twist_body = F_current.M.Inverse() * twist_rcm
        return np.array([
            twist_body.vel.x(), twist_body.vel.y(), twist_body.vel.z(),
            twist_body.rot.x(), twist_body.rot.y(), twist_body.rot.z()
        ])

    def vector_to_twist_msg(self, vec):
        t = Twist()
        t.linear.x = clamp(vec[0], -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
        t.linear.y = clamp(vec[1], -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
        t.linear.z = clamp(vec[2], -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
        t.angular.x = clamp(vec[3], -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
        t.angular.y = clamp(vec[4], -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
        t.angular.z = clamp(vec[5], -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
        return t

    def calculate_initial_rcm(self, q, gripper_transform, rcm_dist):
        frame_j6 = kdl.Frame()
        self.fk_solver.JntToCart(q, frame_j6)
        frame_gripper = frame_j6 * gripper_transform 
        rcm_point = frame_gripper * kdl.Vector(-rcm_dist, 0, 0)
        return kdl.Frame(frame_gripper.M, rcm_point)

    # ----------------------------------------------------------------------
    # [핵심 변경] Hybrid Blending RCM Velocity Solver
    # Position PID 출력(Linear Error)도 회전을 유발하도록 설계됨
    # ----------------------------------------------------------------------
    # def solve_rcm_velocity(self, q, twist_msg, gripper_transform, rcm_point_base):
    #     # 1. FK & Jacobian
    #     frame_j6 = kdl.Frame()
    #     self.fk_solver.JntToCart(q, frame_j6)
    #     frame_gripper = frame_j6 * gripper_transform
        
    #     jac_j6 = kdl.Jacobian(6)
    #     self.jac_solver.JntToJac(q, jac_j6)
    #     offset_j6_grip = frame_j6.M * gripper_transform.p 
    #     jac_j6.changeRefPoint(offset_j6_grip)
        
    #     J_base = np.zeros((6, 6))
    #     for i in range(6):
    #         for j in range(6): J_base[i, j] = jac_j6[i, j]

    #     # 2. Input Parsing (Twist represents PID Error Correction Velocity)
    #     R_base_grip = frame_gripper.M
        
    #     # Linear PID Output -> Base Frame
    #     v_ins_body = kdl.Vector(twist_msg.linear.x, 0, 0) # Insertion Correction
    #     v_ins_base = R_base_grip * v_ins_body
        
    #     v_pos_err_body = kdl.Vector(0, twist_msg.linear.y, twist_msg.linear.z) # Lateral Position Correction
    #     v_pos_err_base = R_base_grip * v_pos_err_body

    #     # Angular PID Output -> Base Frame
    #     w_roll_body = kdl.Vector(twist_msg.angular.x, 0, 0) # Roll Correction
    #     w_roll_base = R_base_grip * w_roll_body
        
    #     w_rot_err_body = kdl.Vector(0, twist_msg.angular.y, twist_msg.angular.z) # Orientation Correction
    #     w_rot_err_base = R_base_grip * w_rot_err_body
        
    #     # 3. Blending Logic
    #     r_rcm_to_grip = frame_gripper.p - rcm_point_base
    #     dist_sq = kdl.dot(r_rcm_to_grip, r_rcm_to_grip)
    #     if dist_sq < 1e-6: dist_sq = 1e-6
        
    #     # (A) Linear PID가 유발하는 회전 (위치 오차를 줄이기 위한 피벗)
    #     w_induced_from_linear = (r_rcm_to_grip * v_pos_err_base) / dist_sq
        
    #     # (B) Angular PID가 유발하는 회전 (각도 오차를 줄이기 위한 피벗)
    #     w_direct_from_angular = w_rot_err_base
        
    #     # (C) 가중합
    #     w_pivot_total = (w_induced_from_linear * WEIGHT_LINEAR_ERROR) + \
    #                     (w_direct_from_angular * WEIGHT_ANGULAR_ERROR)

    #     # 4. Reconstruct Twist consistent with RCM
    #     v_pivot_result_base = w_pivot_total * r_rcm_to_grip
        
    #     # RCM Correction
    #     axis_shaft_base = R_base_grip.UnitX()
    #     vec_grip_rcm = rcm_point_base - frame_gripper.p
    #     projection_length = kdl.dot(vec_grip_rcm, axis_shaft_base) 
    #     closest_point = frame_gripper.p + axis_shaft_base * projection_length
    #     error_vec = rcm_point_base - closest_point
    #     v_correction_base = error_vec * RCM_GAIN_P
        
    #     # Total Velocity
    #     v_total_base = v_ins_base + v_pivot_result_base + v_correction_base
    #     w_total_base = w_roll_base + w_pivot_total
        
    #     V_cmd_solver = np.array([
    #         v_total_base.x(), v_total_base.y(), v_total_base.z(), 
    #         w_total_base.x(), w_total_base.y(), w_total_base.z()
    #     ])
        
    #     # 5. Solve IK
    #     damping = 0.02 
    #     A = np.dot(J_base, J_base.T) + (damping**2) * np.eye(6)
    #     x = np.linalg.solve(A, V_cmd_solver)
    #     return np.dot(J_base.T, x)
    

# ----------------------------------------------------------------------
    # [Fixed] Decoupled RCM Velocity Solver
    # Pivot is driven by Linear Error. Roll is driven by Angular Error.
    # ----------------------------------------------------------------------
    def solve_rcm_velocity(self, q, twist_msg, gripper_transform, rcm_point_base):
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

        # 2. Vector Basis Setup
        R_base_grip = frame_gripper.M
        r_rcm_to_grip = frame_gripper.p - rcm_point_base
        dist_sq = kdl.dot(r_rcm_to_grip, r_rcm_to_grip)
        if dist_sq < 1e-6: dist_sq = 1e-6

        # --- DECOUPLING LOGIC ---
        
        # A. Insertion (Linear X in Body Frame)
        v_ins_body = kdl.Vector(twist_msg.linear.x, 0, 0)
        v_ins_base = R_base_grip * v_ins_body
        
        # B. Roll (Angular X in Body Frame) - Driven by Angular PID
        w_roll_body = kdl.Vector(twist_msg.angular.x, 0, 0)
        w_roll_base = R_base_grip * w_roll_body

        # C. Pivot (Pitch/Yaw) - Driven by LINEAR PID (Position Error)
        # We calculate the velocity needed to move the tip laterally
        v_pos_err_body = kdl.Vector(0, twist_msg.linear.y, twist_msg.linear.z)
        v_pos_err_base = R_base_grip * v_pos_err_body
        
        # Calculate the rotation (w) required to achieve this linear velocity (v)
        # w = (r x v) / r^2
        w_pivot_induced = (r_rcm_to_grip * v_pos_err_base) / dist_sq
        
        # Re-calculate the linear velocity consistent with this pivot 
        # (This ensures v and w are perfectly orthogonal)
        v_pivot_clean = w_pivot_induced * r_rcm_to_grip

        # D. RCM Constraint Correction
        axis_shaft_base = R_base_grip.UnitX()
        vec_grip_rcm = rcm_point_base - frame_gripper.p
        projection_length = kdl.dot(vec_grip_rcm, axis_shaft_base) 
        closest_point = frame_gripper.p + axis_shaft_base * projection_length
        error_vec = rcm_point_base - closest_point
        v_correction_base = error_vec * RCM_GAIN_P

        # 3. Total Velocity Synthesis
        # Linear = Insertion + Pivot(from Position) + RCM Correction
        v_total_base = v_ins_base + v_pivot_clean + v_correction_base
        
        # Angular = Roll(from Orientation) + Pivot(from Position)
        # Note: We IGNORE twist_msg.angular.y/z here to prevent fighting
        w_total_base = w_roll_base + w_pivot_induced
        
        V_cmd_solver = np.array([
            v_total_base.x(), v_total_base.y(), v_total_base.z(), 
            w_total_base.x(), w_total_base.y(), w_total_base.z()
        ])
        
        # 4. Solve IK with Damping
        damping = 0.02 
        A = np.dot(J_base, J_base.T) + (damping**2) * np.eye(6)
        x = np.linalg.solve(A, V_cmd_solver)
        return np.dot(J_base.T, x)

    def publish_joint_vel(self, q_dot, publisher):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.velocity = q_dot.flatten().tolist()
        publisher.publish(msg)

    def publish_gripper_wrt_rcm(self, q, grip_trans, rcm_frame_base, publisher):
        pose = self.kdl_frame_to_pose_msg(self.get_current_pose_wrt_rcm(q, grip_trans, rcm_frame_base))
        publisher.publish(pose)

    def publish_relative_poses(self):
        if self.left_q is None or self.right_q is None: return
        fk_l, fk_r = kdl.Frame(), kdl.Frame()
        self.fk_solver.JntToCart(self.left_q, fk_l)
        self.fk_solver.JntToCart(self.right_q, fk_r)
        
        T_BaseL_GripL = fk_l * self.left_gripper_frame_kdl
        T_BaseR_GripR = fk_r * self.right_gripper_frame_kdl
        T_BaseL_GripR = self.T_LeftBase_RightBase * T_BaseR_GripR
        T_GripR_GripL = T_BaseL_GripR.Inverse() * T_BaseL_GripL
        
        self.pub_left_wrt_right.publish(self.kdl_frame_to_pose_msg(T_GripR_GripL))
        self.pub_right_wrt_left.publish(self.kdl_frame_to_pose_msg(T_GripR_GripL.Inverse()))

def main(args=None):
    rclpy.init(args=args)
    node = RCMTwoFR5HighFreqController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()