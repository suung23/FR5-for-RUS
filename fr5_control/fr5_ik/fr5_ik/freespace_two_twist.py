"""
Freespace Control of two fr5s using Twist input.
Standard Diff-IK controller: Input twist moves the gripper in 6-DOF.
(No RCM/Trocar constraint).

Node inputs:
/fr5_left/desired_twist <Twist>: Desired velocity in Gripper Frame (Linear + Angular)
/fr5_left/joint_states <JointState> : Angles of the joints. In radians

/fr5_right/desired_twist
/fr5_right/joint_states

Node outputs:
/fr5_left/joint_velocity_cmds <JointState> : calculated velocity of the joints in rad/s
/fr5_left/gripper_wrt_rcm <Pose>: Pose of the gripper frame with respect to the Reference Frame (Initial RCM Frame)

/fr5_right/joint_velocity_cmds
/fr5_right/gripper_wrt_rcm

/fr5/left_gripper_wrt_right_gripper <Pose> : Pose of the left gripper frame with respect to the right gripper frame
/fr5/right_gripper_wrt_left_gripper
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Pose
import PyKDL as kdl
import numpy as np
import math

### Constants ###
JOINT_TIMEOUT_SEC = 0.2  # Safety timeout for joint states

#Transformation: J6 -> Gripper
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

def numpy_to_kdl_frame(mat):
    rot = kdl.Rotation(
        mat[0,0], mat[0,1], mat[0,2],
        mat[1,0], mat[1,1], mat[1,2],
        mat[2,0], mat[2,1], mat[2,2]
    )
    vec = kdl.Vector(mat[0,3], mat[1,3], mat[2,3])
    return kdl.Frame(rot, vec)

def kdl_frame_to_pose_msg(frame):
    p = Pose()
    p.position.x = frame.p.x()
    p.position.y = frame.p.y()
    p.position.z = frame.p.z()
    x, y, z, w = frame.M.GetQuaternion()
    p.orientation.x = x
    p.orientation.y = y
    p.orientation.z = z
    p.orientation.w = w
    return p

class FreespaceTwoFR5TwistController(Node):
    def __init__(self):
        super().__init__('freespace_two_fr5_twist_controller')
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        
        # --- Kinematics Setup ---
        self.chain = self.build_fr5_chain()
        
        self.left_gripper_frame_kdl = numpy_to_kdl_frame(FR5_LEFT_GRIPPER_WRT_J6_TRANS)
        self.right_gripper_frame_kdl = numpy_to_kdl_frame(FR5_RIGHT_GRIPPER_WRT_J6_TRANS)
        
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)

        # --- State Variables ---
        self.left_q = None
        self.right_q = None
        
        # Timestamps for safety check
        self.last_left_joint_time = None
        self.last_right_joint_time = None
        
        # Stores kdl.Frame (Reference Frame)
        self.left_ref_frame_base = None 
        self.right_ref_frame_base = None
        
        # --- Publishers & Subscribers ---
        # LEFT ARM
        self.sub_left_joints = self.create_subscription(
            JointState, '/fr5_left/joint_states', self.left_joint_cb, 10)
        self.sub_left_twist = self.create_subscription(
            Twist, '/fr5_left/desired_twist', self.left_twist_cb, 10)
        self.pub_left_vel = self.create_publisher(
            JointState, '/fr5_left/joint_velocity_cmds', 10)
        self.pub_left_rcm_pose = self.create_publisher(
            Pose, '/fr5_left/gripper_wrt_rcm', 10)
            
        # RIGHT ARM
        self.sub_right_joints = self.create_subscription(
            JointState, '/fr5_right/joint_states', self.right_joint_cb, 10)
        self.sub_right_twist = self.create_subscription(
            Twist, '/fr5_right/desired_twist', self.right_twist_cb, 10)
        self.pub_right_vel = self.create_publisher(
            JointState, '/fr5_right/joint_velocity_cmds', 10)
        self.pub_right_rcm_pose = self.create_publisher(
            Pose, '/fr5_right/gripper_wrt_rcm', 10)

        # RELATIVE POSES
        self.pub_left_wrt_right = self.create_publisher(
            Pose, '/fr5/left_gripper_wrt_right_gripper', 10)
        self.pub_right_wrt_left = self.create_publisher(
            Pose, '/fr5/right_gripper_wrt_left_gripper', 10)

        # --- Base Transformations ---
        d = FR5_BASE_DISTANCE
        angle = np.radians(135) # 90 + 45 (-x, +y)
        dx = d * np.cos(angle)
        dy = d * np.sin(angle)
        self.T_LeftBase_RightBase = kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(dx, dy, 0))

    def build_fr5_chain(self):
        chain = kdl.Chain()
        chain.addSegment(kdl.Segment("j1", kdl.Joint("j1", kdl.Joint.RotZ),
            kdl.Frame(kdl.Rotation.RPY(1.5708, 0, 0), kdl.Vector(0, 0, 0.152))))
        chain.addSegment(kdl.Segment("j2", kdl.Joint("j2", kdl.Joint.RotZ),
            kdl.Frame(kdl.Rotation.RPY(0, 0, 0), kdl.Vector(-0.425, 0, 0))))
        chain.addSegment(kdl.Segment("j3", kdl.Joint("j3", kdl.Joint.RotZ),
            kdl.Frame(kdl.Rotation.RPY(0, 0, 0), kdl.Vector(-0.39501, 0, 0))))
        chain.addSegment(kdl.Segment("j4", kdl.Joint("j4", kdl.Joint.RotZ),
            kdl.Frame(kdl.Rotation.RPY(1.5708, 0, 0), kdl.Vector(0, 0, 0.1021))))
        chain.addSegment(kdl.Segment("j5", kdl.Joint("j5", kdl.Joint.RotZ),
            kdl.Frame(kdl.Rotation.RPY(-1.5708, 0, 0), kdl.Vector(0, 0, 0.102))))
        chain.addSegment(kdl.Segment("j6", kdl.Joint("j6", kdl.Joint.RotZ),
            kdl.Frame.Identity()))
        return chain

    def get_kdl_jnt_array(self, joint_state_msg):
        q = kdl.JntArray(6)
        if len(joint_state_msg.position) >= 6:
            for i in range(6):
                q[i] = joint_state_msg.position[i]
        return q
    
    def is_joint_state_fresh(self, last_time_point, arm_name=""):
        if last_time_point is None:
            self.get_logger().warn(f"[{arm_name}] No joint states received yet. Skipping.", throttle_duration_sec=1.0)
            return False
            
        time_diff = (self.get_clock().now() - last_time_point).nanoseconds / 1e9
        
        if time_diff > JOINT_TIMEOUT_SEC:
            self.get_logger().warn(
                f"[{arm_name}] Joint state stale! Delay: {time_diff:.3f}s (> {JOINT_TIMEOUT_SEC}s). Stopping IK.",
                throttle_duration_sec=1.0
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Joint Callbacks
    # ------------------------------------------------------------------

    def left_joint_cb(self, msg):
        self.left_q = self.get_kdl_jnt_array(msg)
        self.last_left_joint_time = self.get_clock().now()

        if self.left_ref_frame_base is None:
            self.left_ref_frame_base = self.calculate_initial_ref(
                self.left_q, self.left_gripper_frame_kdl, FR5_LEFT_RCM_GRIPPER_DISTANCE)
            self.get_logger().info(f"Left Reference Frame Initialized")
        
        self.publish_gripper_wrt_ref(self.left_q, self.left_gripper_frame_kdl, self.left_ref_frame_base, self.pub_left_rcm_pose)
        self.publish_relative_poses()

    def right_joint_cb(self, msg):
        self.right_q = self.get_kdl_jnt_array(msg)
        self.last_right_joint_time = self.get_clock().now()
        
        if self.right_ref_frame_base is None:
            self.right_ref_frame_base = self.calculate_initial_ref(
                self.right_q, self.right_gripper_frame_kdl, FR5_RIGHT_RCM_GRIPPER_DISTANCE)
            self.get_logger().info(f"Right Reference Frame Initialized")
        
        self.publish_gripper_wrt_ref(self.right_q, self.right_gripper_frame_kdl, self.right_ref_frame_base, self.pub_right_rcm_pose)
        self.publish_relative_poses()

    # ------------------------------------------------------------------
    # Twist Callbacks (Updated for Freespace)
    # ------------------------------------------------------------------

    def left_twist_cb(self, msg):
        # 1. Check existence
        if self.left_q is None: return
            
        # 2. Check freshness (Safety)
        if not self.is_joint_state_fresh(self.last_left_joint_time, "LEFT"):
            return 

        # 3. Solve (Freespace)
        q_dot = self.solve_freespace_velocity(self.left_q, msg, self.left_gripper_frame_kdl)
        self.publish_joint_vel(q_dot, self.pub_left_vel)

    def right_twist_cb(self, msg):
        # 1. Check existence
        if self.right_q is None: return
            
        # 2. Check freshness (Safety)
        if not self.is_joint_state_fresh(self.last_right_joint_time, "RIGHT"):
            return 
            
        # 3. Solve (Freespace)
        q_dot = self.solve_freespace_velocity(self.right_q, msg, self.right_gripper_frame_kdl)
        self.publish_joint_vel(q_dot, self.pub_right_vel)

    # ------------------------------------------------------------------
    # Core Logic
    # ------------------------------------------------------------------

    def calculate_initial_ref(self, q, gripper_transform, dist):
        """
        Calculates an arbitrary Reference Frame to keep 'gripper_wrt_rcm' topic valid.
        """
        frame_j6 = kdl.Frame()
        self.fk_solver.JntToCart(q, frame_j6)
        frame_gripper = frame_j6 * gripper_transform 
        
        # Just create a point 'dist' away to mimic RCM frame, used as World Origin for relative topic
        ref_point = frame_gripper * kdl.Vector(-dist, 0, 0)
        ref_rot = frame_gripper.M
        return kdl.Frame(ref_rot, ref_point)

    def solve_freespace_velocity(self, q, twist_msg, gripper_transform):
        """
        Solves Diff-IK for Freespace (6-DOF) motion.
        INPUT: twist_msg in GRIPPER (Body) Frame.
        """
        # 1. FK to get Orientation (Base -> Gripper)
        frame_j6 = kdl.Frame()
        self.fk_solver.JntToCart(q, frame_j6)
        frame_gripper = frame_j6 * gripper_transform
        
        # 2. Jacobian at Gripper Origin
        jac_j6 = kdl.Jacobian(6)
        self.jac_solver.JntToJac(q, jac_j6)
        offset_j6_grip = frame_j6.M * gripper_transform.p 
        jac_j6.changeRefPoint(offset_j6_grip)
        
        J_base = np.zeros((6, 6))
        for i in range(6):
            for j in range(6): J_base[i, j] = jac_j6[i, j]

        # 3. Convert Input Twist (Body) -> Base Frame
        R_base_grip = frame_gripper.M

        # Linear Velocity
        v_cmd_body = kdl.Vector(twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z)
        v_cmd_base = R_base_grip * v_cmd_body

        # Angular Velocity
        w_cmd_body = kdl.Vector(twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z)
        w_cmd_base = R_base_grip * w_cmd_body
        
        # 4. Construct Target Twist Vector (Base Frame)
        V_cmd_base = np.array([
            v_cmd_base.x(), v_cmd_base.y(), v_cmd_base.z(),
            w_cmd_base.x(), w_cmd_base.y(), w_cmd_base.z()
        ])
        
        # 5. Damped Least Squares (DLS) Solver
        # q_dot = J_dagger * V
        damping = 0.02 
        lambda_sq = damping ** 2

        A = np.dot(J_base, J_base.T) + lambda_sq * np.eye(6)
        x = np.linalg.solve(A, V_cmd_base)
        q_dot_sol = np.dot(J_base.T, x)
        
        return q_dot_sol

    def publish_joint_vel(self, q_dot, publisher):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.velocity = q_dot.flatten().tolist()
        publisher.publish(msg)

    def publish_gripper_wrt_ref(self, q, grip_trans, ref_frame_base, publisher):
        frame_j6 = kdl.Frame()
        self.fk_solver.JntToCart(q, frame_j6)
        frame_gripper_base = frame_j6 * grip_trans
        # Publish pose relative to our arbitrary start frame
        frame_wrt_ref = ref_frame_base.Inverse() * frame_gripper_base
        publisher.publish(kdl_frame_to_pose_msg(frame_wrt_ref))

    def publish_relative_poses(self):
        if self.left_q is None or self.right_q is None: return
        fk_l, fk_r = kdl.Frame(), kdl.Frame()
        self.fk_solver.JntToCart(self.left_q, fk_l)
        self.fk_solver.JntToCart(self.right_q, fk_r)
        
        T_BaseL_GripL = fk_l * self.left_gripper_frame_kdl
        T_BaseR_GripR = fk_r * self.right_gripper_frame_kdl
        T_BaseL_GripR = self.T_LeftBase_RightBase * T_BaseR_GripR
        
        T_GripR_GripL = T_BaseL_GripR.Inverse() * T_BaseL_GripL
        T_GripL_GripR = T_GripR_GripL.Inverse()
        
        self.pub_left_wrt_right.publish(kdl_frame_to_pose_msg(T_GripR_GripL))
        self.pub_right_wrt_left.publish(kdl_frame_to_pose_msg(T_GripL_GripR))

def main(args=None):
    rclpy.init(args=args)
    node = FreespaceTwoFR5TwistController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()