"""
Freespace Position Controller for two FR5s (No RCM Constraint).
Designed for Imitation Learning (ACT, Diffusion Policy).

Inputs:
/fr5_left/desired_pose_wrt_rcm <Pose>: Desired Position of the gripper with respect to the RCM frame (Reference Frame)
/fr5_left/joint_states <JointState> : Angles of the joints. In radians

/fr5_right/desired_pose_wrt_rcm
/fr5_right/joint_states

Outputs:
/fr5_left/joint_velocity_cmds <JointState> : calculated velocity of the joints in rad/s
/fr5_left/current_gripper_wrt_rcm <Pose>: Pose of the gripper frame with respect to the RCM frame 

/fr5_right/joint_velocity_cmds
/fr5_right/current_gripper_wrt_rcm

/fr5/left_gripper_wrt_right_gripper <Pose> : Pose of the left gripper frame with respect to the right gripper frame
/fr5/right_gripper_wrt_left_gripper

Updated:
- Removed RCM kinematic constraints (Pivot logic).
- Now uses standard Diff-IK for 6-DOF free motion relative to the reference frame.
- High-frequency (100Hz) PID Control Loop.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Pose
import PyKDL as kdl
import numpy as np
import math

### PID Constants (Freespace Tuning) ###
# Freespace motion can be slightly more aggressive than constrained motion
PID_KP_POS = 3.0
PID_KI_POS = 0.01
PID_KD_POS = 0.1

PID_KP_ROT = 2.0
PID_KI_ROT = 0.01
PID_KD_ROT = 0.05

MAX_I_TERM = 0.05
MAX_LINEAR_VEL = 0.25   # Increased slightly for freespace
MAX_ANGULAR_VEL = 1.5
JOINT_TIMEOUT_SEC = 0.2
CONTROL_RATE_HZ = 100.0

# --- Transformation Constants ---
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

        # P Term
        p_term = np.zeros(6)
        p_term[:3] = error_vector[:3] * PID_KP_POS
        p_term[3:] = error_vector[3:] * PID_KP_ROT

        # I Term
        self.integral[:3] += error_vector[:3] * dt
        self.integral[3:] += error_vector[3:] * dt
        self.integral = np.clip(self.integral, -MAX_I_TERM, MAX_I_TERM)
        
        i_term = np.zeros(6)
        i_term[:3] = self.integral[:3] * PID_KI_POS
        i_term[3:] = self.integral[3:] * PID_KI_ROT

        # D Term
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


class FreespaceTwoFR5Controller(Node):
    def __init__(self):
        super().__init__('freespace_two_fr5_controller')
        self.chain = self.build_fr5_chain()
        
        self.left_gripper_frame_kdl = self.numpy_to_kdl_frame(FR5_LEFT_GRIPPER_WRT_J6_TRANS)
        self.right_gripper_frame_kdl = self.numpy_to_kdl_frame(FR5_RIGHT_GRIPPER_WRT_J6_TRANS)
        
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)

        # PID
        self.pid_left = PIDController6D()
        self.pid_right = PIDController6D()
        
        # State
        self.left_q = None
        self.right_q = None
        self.left_des_pose_msg = None
        self.right_des_pose_msg = None

        self.last_left_joint_time = None
        self.last_right_joint_time = None
        self.last_control_time = self.get_clock().now()
        
        # Reference Frames (Still named "RCM" for compatibility, but acts as World/Task Frame)
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

        # Base Trans
        d = FR5_BASE_DISTANCE
        angle = np.radians(135)
        self.T_LeftBase_RightBase = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(d*np.cos(angle), d*np.sin(angle), 0))

        # Timer
        self.timer = self.create_timer(1.0 / CONTROL_RATE_HZ, self.control_loop)
        self.get_logger().info(f"Freespace Controller Started at {CONTROL_RATE_HZ}Hz")

    # --- Helpers ---
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
            self.left_rcm_frame_base = self.calculate_initial_reference_frame(self.left_q, self.left_gripper_frame_kdl, FR5_LEFT_RCM_GRIPPER_DISTANCE)
            self.get_logger().info("Left Reference Frame Initialized")
        self.publish_gripper_wrt_ref(self.left_q, self.left_gripper_frame_kdl, self.left_rcm_frame_base, self.pub_left_rcm_pose)
        self.publish_relative_poses()

    def right_joint_cb(self, msg):
        self.right_q = self.get_kdl_jnt_array(msg)
        self.last_right_joint_time = self.get_clock().now()
        if self.right_rcm_frame_base is None:
            self.right_rcm_frame_base = self.calculate_initial_reference_frame(self.right_q, self.right_gripper_frame_kdl, FR5_RIGHT_RCM_GRIPPER_DISTANCE)
            self.get_logger().info("Right Reference Frame Initialized")
        self.publish_gripper_wrt_ref(self.right_q, self.right_gripper_frame_kdl, self.right_rcm_frame_base, self.pub_right_rcm_pose)
        self.publish_relative_poses()

    def left_pose_cb(self, msg):
        self.left_des_pose_msg = msg

    def right_pose_cb(self, msg):
        self.right_des_pose_msg = msg

    # --- Control Loop ---
    def control_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_control_time).nanoseconds / 1e9
        self.last_control_time = now
        if dt > 0.1: dt = 0.01

        # Left
        if self.check_ready(self.left_q, self.left_des_pose_msg, self.left_rcm_frame_base, self.last_left_joint_time, "LEFT"):
            F_des = self.pose_msg_to_kdl_frame(self.left_des_pose_msg)
            F_curr = self.get_current_pose_wrt_ref(self.left_q, self.left_gripper_frame_kdl, self.left_rcm_frame_base)
            
            error_vec = self.compute_error_vector(F_curr, F_des) # Error in Body Frame
            pid_out = self.pid_left.compute(error_vec, dt)       # Twist in Body Frame
            
            twist_cmd = self.vector_to_twist_msg(pid_out)
            q_dot = self.solve_freespace_velocity(self.left_q, twist_cmd, self.left_gripper_frame_kdl)
            self.publish_joint_vel(q_dot, self.pub_left_vel)

        # Right
        if self.check_ready(self.right_q, self.right_des_pose_msg, self.right_rcm_frame_base, self.last_right_joint_time, "RIGHT"):
            F_des = self.pose_msg_to_kdl_frame(self.right_des_pose_msg)
            F_curr = self.get_current_pose_wrt_ref(self.right_q, self.right_gripper_frame_kdl, self.right_rcm_frame_base)
            
            error_vec = self.compute_error_vector(F_curr, F_des)
            pid_out = self.pid_right.compute(error_vec, dt)
            
            twist_cmd = self.vector_to_twist_msg(pid_out)
            q_dot = self.solve_freespace_velocity(self.right_q, twist_cmd, self.right_gripper_frame_kdl)
            self.publish_joint_vel(q_dot, self.pub_right_vel)

    # --- Math & Solvers ---
    def solve_freespace_velocity(self, q, twist_body_msg, gripper_transform):
        """
        Solves Diff-IK for Freespace motion.
        Converts Body-Frame Twist (from PID) -> Base-Frame Twist -> Joint Velocities.
        """
        # 1. FK to get current orientation (Base -> Gripper)
        frame_j6 = kdl.Frame()
        self.fk_solver.JntToCart(q, frame_j6)
        frame_gripper = frame_j6 * gripper_transform
        
        # 2. Jacobian Calculation
        jac_j6 = kdl.Jacobian(6)
        self.jac_solver.JntToJac(q, jac_j6)
        # Shift Jacobian reference point to Gripper (Body) Origin
        offset_j6_grip = frame_j6.M * gripper_transform.p 
        jac_j6.changeRefPoint(offset_j6_grip)

        # 3. Transform PID Output (Body Frame) -> Base Frame
        # Because Jacobian relates q_dot to Twist in Base Frame
        R_base_grip = frame_gripper.M
        
        v_body = kdl.Vector(twist_body_msg.linear.x, twist_body_msg.linear.y, twist_body_msg.linear.z)
        v_base = R_base_grip * v_body
        
        w_body = kdl.Vector(twist_body_msg.angular.x, twist_body_msg.angular.y, twist_body_msg.angular.z)
        w_base = R_base_grip * w_body
        
        V_target_base = np.array([
            v_base.x(), v_base.y(), v_base.z(),
            w_base.x(), w_base.y(), w_base.z()
        ])
        
        # 4. Damped Least Squares Solver: q_dot = J# * V
        J = np.zeros((6, 6))
        for i in range(6):
            for j in range(6): J[i, j] = jac_j6[i, j]

        damping = 0.02
        lambda_sq = damping ** 2
        # (J * J.T + lambda^2 * I) * x = V
        A = np.dot(J, J.T) + lambda_sq * np.eye(6)
        x = np.linalg.solve(A, V_target_base)
        q_dot = np.dot(J.T, x)
        
        return q_dot

    def check_ready(self, q, des_pose, ref_frame, last_time, name):
        if q is None or des_pose is None or ref_frame is None: return False
        if last_time is None: return False
        if (self.get_clock().now() - last_time).nanoseconds / 1e9 > JOINT_TIMEOUT_SEC:
            self.get_logger().warn(f"[{name}] Stale joints!", throttle_duration_sec=1.0)
            return False
        return True

    def get_current_pose_wrt_ref(self, q, grip_trans, ref_frame):
        fk = kdl.Frame()
        self.fk_solver.JntToCart(q, fk)
        # T_ref_grip = inv(T_base_ref) * T_base_grip
        return ref_frame.Inverse() * (fk * grip_trans)

    def compute_error_vector(self, F_current, F_desired):
        twist_ref = kdl.diff(F_current, F_desired)
        twist_body = F_current.M.Inverse() * twist_ref
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

    def calculate_initial_reference_frame(self, q, gripper_transform, dist):
        # We keep the "RCM" style initialization just to have a consistent start frame
        # logic, but in freespace mode, this is just an arbitrary zero point in space.
        frame_j6 = kdl.Frame()
        self.fk_solver.JntToCart(q, frame_j6)
        frame_gripper = frame_j6 * gripper_transform 
        ref_point = frame_gripper * kdl.Vector(-dist, 0, 0)
        return kdl.Frame(frame_gripper.M, ref_point)

    def publish_joint_vel(self, q_dot, publisher):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.velocity = q_dot.flatten().tolist()
        publisher.publish(msg)

    def publish_gripper_wrt_ref(self, q, grip_trans, ref_frame, publisher):
        pose = self.kdl_frame_to_pose_msg(self.get_current_pose_wrt_ref(q, grip_trans, ref_frame))
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
    node = FreespaceTwoFR5Controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()