"""
Publisher for Gripper State w.r.t a newly defined RCM frame.
State consists of 4 values:
[0:3] Position (x, y, z) of the gripper w.r.t the new RCM frame (parallel to Space Frame)
[3]   Roll angle of the laparoscope tool around its X-axis, relative to the Space XY plane.

Space Frame is defined as rotated 5/4 pi around Z-axis w.r.t Base Frame.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
import PyKDL as kdl
import numpy as np
import math

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

def numpy_to_kdl_frame(mat):
    rot = kdl.Rotation(
        mat[0,0], mat[0,1], mat[0,2],
        mat[1,0], mat[1,1], mat[1,2],
        mat[2,0], mat[2,1], mat[2,2]
    )
    vec = kdl.Vector(mat[0,3], mat[1,3], mat[2,3])
    return kdl.Frame(rot, vec)

class GripperStatePublisher(Node):
    def __init__(self):
        super().__init__('calc_gripper_wrt_new_rcm_node')
        
        # --- Kinematics Setup ---
        self.chain = self.build_fr5_chain()
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        
        self.left_gripper_frame_kdl = numpy_to_kdl_frame(FR5_LEFT_GRIPPER_WRT_J6_TRANS)
        self.right_gripper_frame_kdl = numpy_to_kdl_frame(FR5_RIGHT_GRIPPER_WRT_J6_TRANS)
        
        # --- Space Frame Definition ---
        # Base frame에서 Z축으로 5/4 pi 회전한 프레임
        self.R_space_base = kdl.Rotation.RotZ(5.0 * math.pi / 4.0)
        self.R_base_space = self.R_space_base.Inverse() # Base벡터를 Space로 변환할 때 사용
        
        # --- State Variables ---
        self.left_rcm_pos_base = None   # Base 프레임 기준의 RCM 위치 (kdl.Vector)
        self.right_rcm_pos_base = None  
        
        # --- Publishers & Subscribers ---
        self.sub_left_joints = self.create_subscription(
            JointState, '/fr5_left/joint_states', self.left_joint_cb, 10)
        self.pub_left_state = self.create_publisher(
            Float32MultiArray, '/fr5_left/gripper_wrt_new_rcm', 10)
            
        self.sub_right_joints = self.create_subscription(
            JointState, '/fr5_right/joint_states', self.right_joint_cb, 10)
        self.pub_right_state = self.create_publisher(
            Float32MultiArray, '/fr5_right/gripper_wrt_new_rcm', 10)

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

    def calculate_initial_rcm_pos(self, q, gripper_transform, rcm_dist):
        """
        초기 RCM 위치 계산 (Base Frame 기준)
        방향은 고려하지 않고 고정점(Point)의 위치만 반환함
        """
        frame_j6 = kdl.Frame()
        self.fk_solver.JntToCart(q, frame_j6)
        frame_gripper = frame_j6 * gripper_transform 
        rcm_point = frame_gripper * kdl.Vector(-rcm_dist, 0, 0)
        return rcm_point

    def calculate_4d_state(self, q, gripper_transform, rcm_pos_base):
        """
        주어진 조건에 따라 4개의 값을 계산하여 반환
        [x, y, z, roll]
        """
        # 1. Forward Kinematics
        frame_j6 = kdl.Frame()
        self.fk_solver.JntToCart(q, frame_j6)
        frame_gripper_base = frame_j6 * gripper_transform
        
        # 2. 새로운 RCM 프레임(Space Frame과 평행)에 대한 그리퍼 위치 계산
        vec_rcm_to_g_base = frame_gripper_base.p - rcm_pos_base
        # Base 기준 벡터를 Space Frame 기준으로 회전
        pos_new_rcm = self.R_base_space * vec_rcm_to_g_base
        
        # 3. Tool Roll 계산
        # 그리퍼의 X, Y축 벡터를 Space Frame 기준으로 변환
        X_g_base = frame_gripper_base.M.UnitX()
        Y_g_base = frame_gripper_base.M.UnitY()
        
        X_g_space = self.R_base_space * X_g_base
        Y_g_space = self.R_base_space * Y_g_base
        
        # Space Frame의 XY 평면과 평행하고 기구의 X축에 직교하는 '기준 Y축(Y_a)' 생성
        Z_space = kdl.Vector(0, 0, 1)
        Y_a_unnorm = Z_space * X_g_space # 외적 (PyKDL에서는 * 연산자가 외적)
        
        roll = 0.0
        # Singularity 방지 (기구가 완벽하게 Z축 방향으로 서있을 경우)
        if Y_a_unnorm.Norm() > 1e-6:
            Y_a = Y_a_unnorm / Y_a_unnorm.Norm()
            
            # X_g_space 축을 기준으로 Y_a 벡터가 Y_g_space 벡터로 얼마나 회전했는지 각도 계산
            cos_roll = kdl.dot(Y_a, Y_g_space)
            sin_roll = kdl.dot(Y_a * Y_g_space, X_g_space) # (Y_a x Y_g) 내적 X_g
            roll = math.atan2(sin_roll, cos_roll)

        return [pos_new_rcm.x(), pos_new_rcm.y(), pos_new_rcm.z(), roll]

    def left_joint_cb(self, msg):
        q = self.get_kdl_jnt_array(msg)
        
        # 처음 한 번만 RCM 위치 고정
        if self.left_rcm_pos_base is None:
            self.left_rcm_pos_base = self.calculate_initial_rcm_pos(
                q, self.left_gripper_frame_kdl, FR5_LEFT_RCM_GRIPPER_DISTANCE)
            self.get_logger().info(f"Left RCM Position Initialized: {self.left_rcm_pos_base}")
            
        # 4D 상태 계산 및 퍼블리시
        state = self.calculate_4d_state(q, self.left_gripper_frame_kdl, self.left_rcm_pos_base)
        
        msg_out = Float64MultiArray()
        msg_out.data = state
        self.pub_left_state.publish(msg_out)

    def right_joint_cb(self, msg):
        q = self.get_kdl_jnt_array(msg)
        
        # 처음 한 번만 RCM 위치 고정
        if self.right_rcm_pos_base is None:
            self.right_rcm_pos_base = self.calculate_initial_rcm_pos(
                q, self.right_gripper_frame_kdl, FR5_RIGHT_RCM_GRIPPER_DISTANCE)
            self.get_logger().info(f"Right RCM Position Initialized: {self.right_rcm_pos_base}")
            
        # 4D 상태 계산 및 퍼블리시
        state = self.calculate_4d_state(q, self.right_gripper_frame_kdl, self.right_rcm_pos_base)
        
        msg_out = Float64MultiArray()
        msg_out.data = state
        self.pub_right_state.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = GripperStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()