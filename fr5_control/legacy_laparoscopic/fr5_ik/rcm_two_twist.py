"""
Control of two fr5s using RCM and twist input.
Input corresponds to the twist of the gripper with respect to gripper frame.
Gripper and RCM frame's X-axis is the insertion axis.

Node inputs:
/fr5_left/desired_twist <Twist>: Desired twist of the gripper frame with respect to gripper frame 
/fr5_left/joint_states <JointState> : Angles of the joints. In radians

/fr5_right/desired_twist
/fr5_right/joint_states

Node outputs:
/fr5_left/joint_velocity_cmds <JointState> : calculated velocity of the joints in rad/s
/fr5_left/gripper_wrt_rcm <Pose>: Pose of the gripper frame with respect to the RCM frame 

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
RCM_GAIN_P = 3.0 
JOINT_TIMEOUT_SEC = 0.2  # Safety timeout for joint states
# 입력 가중치 (1.0이면 100% 반영, 0.5면 50% 감쇠)
# 두 입력을 섞어서 사용할 때 감도 조절용입니다.
WEIGHT_LINEAR_INPUT = 0.33   # 선형 입력(Linear Y, Z)이 만드는 회전의 강도
WEIGHT_ANGULAR_INPUT = 1.0  # 각속도 입력(Angular Y, Z)이 만드는 회전의 강도
ROLL_SENSITIVITY = 2.0        # 롤 입력(Angular X)의 감도 조절용


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

class RCMTwoFR5TwistController(Node):
    def __init__(self):
        super().__init__('RCM_two_FR5_twist_controller')
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
        
        # Stores kdl.Frame (Orientation + Position)
        self.left_rcm_frame_base = None 
        self.right_rcm_frame_base = None
        
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
        """
        Checks if the joint state is fresh enough (within JOINT_TIMEOUT_SEC).
        """
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
        self.last_left_joint_time = self.get_clock().now() # Update timestamp

        if self.left_rcm_frame_base is None:
            self.left_rcm_frame_base = self.calculate_initial_rcm(
                self.left_q, self.left_gripper_frame_kdl, FR5_LEFT_RCM_GRIPPER_DISTANCE)
            self.get_logger().info(f"Left RCM Frame Initialized: {self.left_rcm_frame_base.p}")
        
        self.publish_gripper_wrt_rcm(self.left_q, self.left_gripper_frame_kdl, self.left_rcm_frame_base, self.pub_left_rcm_pose)
        self.publish_relative_poses()

    def right_joint_cb(self, msg):
        self.right_q = self.get_kdl_jnt_array(msg)
        self.last_right_joint_time = self.get_clock().now() # Update timestamp
        
        if self.right_rcm_frame_base is None:
            self.right_rcm_frame_base = self.calculate_initial_rcm(
                self.right_q, self.right_gripper_frame_kdl, FR5_RIGHT_RCM_GRIPPER_DISTANCE)
            self.get_logger().info(f"Right RCM Frame Initialized: {self.right_rcm_frame_base.p}")
        
        self.publish_gripper_wrt_rcm(self.right_q, self.right_gripper_frame_kdl, self.right_rcm_frame_base, self.pub_right_rcm_pose)
        self.publish_relative_poses()

    # ------------------------------------------------------------------
    # Twist Callbacks (Updated with Safety Check)
    # ------------------------------------------------------------------

    def left_twist_cb(self, msg):
        # 1. Check existence
        if self.left_q is None or self.left_rcm_frame_base is None: 
            return
            
        # 2. Check freshness (Safety)
        if not self.is_joint_state_fresh(self.last_left_joint_time, "LEFT"):
            return # Stop if stale

        # 3. Solve
        q_dot = self.solve_rcm_velocity(self.left_q, msg, self.left_gripper_frame_kdl, self.left_rcm_frame_base.p)
        self.publish_joint_vel(q_dot, self.pub_left_vel)

    def right_twist_cb(self, msg):
        # 1. Check existence
        if self.right_q is None or self.right_rcm_frame_base is None: 
            return
            
        # 2. Check freshness (Safety)
        if not self.is_joint_state_fresh(self.last_right_joint_time, "RIGHT"):
            return # Stop if stale
            
        # 3. Solve
        q_dot = self.solve_rcm_velocity(self.right_q, msg, self.right_gripper_frame_kdl, self.right_rcm_frame_base.p)
        self.publish_joint_vel(q_dot, self.pub_right_vel)

    # ------------------------------------------------------------------
    # Core Logic
    # ------------------------------------------------------------------

    def calculate_initial_rcm(self, q, gripper_transform, rcm_dist):
        """
        Calculates RCM Frame by projecting along Gripper NEGATIVE X-axis.
        """
        frame_j6 = kdl.Frame()
        self.fk_solver.JntToCart(q, frame_j6)
        frame_gripper = frame_j6 * gripper_transform 
        
        rcm_point = frame_gripper * kdl.Vector(-rcm_dist, 0, 0)
        rcm_rot = frame_gripper.M

        return kdl.Frame(rcm_rot, rcm_point)
    


    def solve_rcm_velocity(self, q, twist_msg, gripper_transform, rcm_point_base):
        """
        [수정됨] Hybrid RCM Control (Linear + Angular Blending)
        INPUT: 
          - twist_msg.linear: X(삽입), Y/Z(피벗 유도)
          - twist_msg.angular: X(롤), Y/Z(직접 피벗 회전)
        """
        # 1. Forward Kinematics
        frame_j6 = kdl.Frame()
        self.fk_solver.JntToCart(q, frame_j6)
        frame_gripper = frame_j6 * gripper_transform
        
        # 2. Jacobian Matrix
        jac_j6 = kdl.Jacobian(6)
        self.jac_solver.JntToJac(q, jac_j6)
        offset_j6_grip = frame_j6.M * gripper_transform.p 
        jac_j6.changeRefPoint(offset_j6_grip)
        
        J_base = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                J_base[i, j] = jac_j6[i, j]

        # 3. Coordinate Transformation (Body -> Base)
        R_base_grip = frame_gripper.M
        
        # --- [Step A] 사용자 입력 분리 ---
        
        # A-1. Linear Input 처리
        # Linear X: 삽입/추출 (RCM과 무관한 독립 모션)
        v_ins_body = kdl.Vector(twist_msg.linear.x, 0, 0)
        v_ins_base = R_base_grip * v_ins_body
        
        # Linear Y, Z: 피벗을 유발하는 선속도
        v_pivot_drive_body = kdl.Vector(0, twist_msg.linear.y, twist_msg.linear.z)
        v_pivot_drive_base = R_base_grip * v_pivot_drive_body

        # A-2. Angular Input 처리
        # Angular X: 롤 회전 (독립 모션)
        w_roll_body = kdl.Vector(twist_msg.angular.x * ROLL_SENSITIVITY, 0, 0)
        w_roll_base = R_base_grip * w_roll_body
        
        # Angular Y, Z: 사용자가 직접 명령한 피벗 회전
        w_pivot_user_body = kdl.Vector(0, twist_msg.angular.y, twist_msg.angular.z)
        w_pivot_user_base = R_base_grip * w_pivot_user_body

        # 4. Enforce RCM Constraint (Calculate Combined Omega)
        
        r_rcm_to_grip = frame_gripper.p - rcm_point_base
        dist_sq = kdl.dot(r_rcm_to_grip, r_rcm_to_grip)
        if dist_sq < 1e-6: dist_sq = 1e-6

        # [핵심 로직] Linear 입력에 의해 유도된 각속도 계산 (w = r x v / r^2)
        # 예: 그리퍼를 오른쪽(Linear Y)으로 밀면, RCM 중심으로 Yaw 회전이 발생해야 함
        w_induced_from_linear = (r_rcm_to_grip * v_pivot_drive_base) / dist_sq
        
        # [Blending] 두 각속도 합치기 (가중합)
        # 최종 피벗 각속도 = (선형 입력 기반 회전 * 가중치) + (각속도 입력 * 가중치)
        w_pivot_total = (w_induced_from_linear * WEIGHT_LINEAR_INPUT) + \
                        (w_pivot_user_base * WEIGHT_ANGULAR_INPUT)

        # 5. Calculate Required Linear Velocity for RCM
        # 이제 결정된 '최종 회전(w_pivot_total)'을 수행하기 위해
        # 그리퍼가 움직여야 하는 선속도(v = w x r)를 다시 계산합니다.
        # 이렇게 해야 Angular 입력도 RCM 제약조건을 위배하지 않고 동작합니다.
        v_pivot_result_base = w_pivot_total * r_rcm_to_grip
        
        # 6. Drift Correction (기존 유지)
        axis_shaft_base = R_base_grip.UnitX()
        vec_grip_rcm = rcm_point_base - frame_gripper.p
        projection_length = kdl.dot(vec_grip_rcm, axis_shaft_base) 
        closest_point_on_shaft = frame_gripper.p + axis_shaft_base * projection_length
        error_vec = rcm_point_base - closest_point_on_shaft
        v_correction_base = error_vec * RCM_GAIN_P
        
        # 7. Final Command Construction
        # 선속도 = 삽입(Input) + 피벗(Result) + 보정(Correction)
        v_total_cmd = v_ins_base + v_pivot_result_base + v_correction_base
        
        # 각속도 = 롤(Input) + 피벗(Total)
        w_total_cmd = w_roll_base + w_pivot_total
        
        V_cmd_solver = np.array([
            v_total_cmd.x(), v_total_cmd.y(), v_total_cmd.z(),
            w_total_cmd.x(), w_total_cmd.y(), w_total_cmd.z()
        ])
        
        # IK Solve (DLS)
        damping = 0.02 
        lambda_sq = damping ** 2
        A = np.dot(J_base, J_base.T) + lambda_sq * np.eye(6)
        x = np.linalg.solve(A, V_cmd_solver)
        q_dot_sol = np.dot(J_base.T, x)
        
        return q_dot_sol
    


    def publish_joint_vel(self, q_dot, publisher):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.velocity = q_dot.flatten().tolist()
        publisher.publish(msg)

    def publish_gripper_wrt_rcm(self, q, grip_trans, rcm_frame_base, publisher):
        frame_j6 = kdl.Frame()
        self.fk_solver.JntToCart(q, frame_j6)
        frame_gripper_base = frame_j6 * grip_trans
        frame_gripper_wrt_rcm = rcm_frame_base.Inverse() * frame_gripper_base
        publisher.publish(kdl_frame_to_pose_msg(frame_gripper_wrt_rcm))

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
    node = RCMTwoFR5TwistController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()