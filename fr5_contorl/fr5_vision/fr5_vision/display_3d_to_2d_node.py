import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Pose
import PyKDL as kdl
import numpy as np
import math
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image

### Constants ###

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

CAMERA_DISTORTION = np.array([
    -0.003664160846246098,
    -0.031178709180679016,
    -2.7559028467925786e-05,
    -0.00017229702522676886,
    0.0059303876279991515
], dtype=np.float32)

CAMERA_K = np.array([
    [470.9939686695459, 0.0, 304.8237579938155],
    [0.0, 472.0387444977319, 199.26806517585842],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# T_Base_Camera (Camera pose in Base Frame)
LEFT_T_BASE_CAMERA = np.array([
        [
            -0.7119189559667939,
            0.6902525369667032,
            -0.12931680225779274,
            -0.7366629195860501
        ],
        [
            0.7015218681368278,
            0.6905576590212005,
            -0.17606018315614264,
            -0.30113244160683805
        ],
        [
            -0.032225279843100994,
            -0.216059146481211,
            -0.9758483368643129,
            0.1898691039190796
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ], dtype=float)

RIGHT_T_BASE_CAMERA = np.array([
        [
            -0.7017756393884957,
            0.699068275559918,
            -0.13716595082800598,
            -0.30043758287096783
        ],
        [
            0.7116427721203145,
            0.6790509808605265,
            -0.18015085423408655,
            -0.7194299714398305
        ],
        [
            -0.032795073559644825,
            -0.22403863840427063,
            -0.974028321791609,
            0.1914547293855008
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ], dtype=float)

CAMERA_TOPIC = '/camera/image1'

# Rod Configuration
ROD_LENGTH = 0.1 
ROD_POINTS_COUNT = 10 


def numpy_to_kdl_frame(mat):
    rot = kdl.Rotation(
        mat[0,0], mat[0,1], mat[0,2],
        mat[1,0], mat[1,1], mat[1,2],
        mat[2,0], mat[2,1], mat[2,2]
    )
    vec = kdl.Vector(mat[0,3], mat[1,3], mat[2,3])
    return kdl.Frame(rot, vec)


def get_pixel_from_robot_point(p_base, T_bc, K, D):
    """
    Maps a 3D point from Robot Base Frame to 2D Pixel coordinates and computes Depth.
    Returns: u, v, depth (in meters)
    """
    # T_cb: Transform from Base Frame to Camera Frame
    T_cb = np.linalg.inv(T_bc)
    
    # 1. Calculate Point in Camera Frame to get Depth (Z-axis)
    if len(p_base) == 3:
        p_base_hom = np.array([p_base[0], p_base[1], p_base[2], 1.0])
    else:
        p_base_hom = np.array(p_base)
        
    p_cam = T_cb @ p_base_hom
    depth = p_cam[2]  # Z-coordinate in camera frame is the metric depth
    
    # 2. Project to Pixel Coordinates
    R_matrix = T_cb[:3, :3]
    t_vec = T_cb[:3, 3]
    
    r_vec, _ = cv2.Rodrigues(R_matrix)
    
    object_points = np.array([p_base[:3]], dtype=np.float32)

    image_points, jacobian = cv2.projectPoints(
        object_points,
        r_vec,
        t_vec,
        K,
        D
    )
    
    uv = image_points[0][0]
    u, v = uv[0], uv[1]
    
    return u, v, depth


class Display3Dto2DNode(Node):
    def __init__(self):
        super().__init__('display_3d_to_2d_node')
        # --- Kinematics Setup ---
        self.chain = self.build_fr5_chain()
        self.left_gripper_frame_kdl = numpy_to_kdl_frame(FR5_LEFT_GRIPPER_WRT_J6_TRANS)
        self.right_gripper_frame_kdl = numpy_to_kdl_frame(FR5_RIGHT_GRIPPER_WRT_J6_TRANS)
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        # --- State Variables ---
        self.left_q = None
        self.right_q = None
        
        self.left_rod_points = []
        self.right_rod_points = []

        self.sub_left_joints = self.create_subscription(
            JointState, '/fr5_left/joint_states', self.left_joint_cb, 10)
        self.sub_right_joints = self.create_subscription(
            JointState, '/fr5_right/joint_states', self.right_joint_cb, 10) 

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.image_callback,
            10
        )

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

    def left_joint_cb(self, msg):
        self.left_q = self.get_kdl_jnt_array(msg)
        self.left_rod_points = self.calculate_rod_points(self.left_q, self.left_gripper_frame_kdl)
    
    def right_joint_cb(self, msg):
        self.right_q = self.get_kdl_jnt_array(msg)
        self.right_rod_points = self.calculate_rod_points(self.right_q, self.right_gripper_frame_kdl)

    def calculate_rod_points(self, q, gripper_transform):
        frame_j6 = kdl.Frame()
        self.fk_solver.JntToCart(q, frame_j6)
        gripper_wrt_base = frame_j6 * gripper_transform 
        points_3d = []
        for i in range(ROD_POINTS_COUNT):
            dist = (ROD_LENGTH / (ROD_POINTS_COUNT - 1)) * i
            point_wrt_base = gripper_wrt_base * kdl.Vector(-dist, 0, 0)
            points_3d.append([point_wrt_base.x(), point_wrt_base.y(), point_wrt_base.z()])
        return points_3d

    def image_callback(self, msg):
        try:
            if not self.left_rod_points or not self.right_rod_points:
                self.get_logger().warn(f"Waiting for joint states...", throttle_duration_sec=2.0)
                return 
            
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # --- DRAW LEFT ROD (BLUE) ---
            for i, point_3d in enumerate(self.left_rod_points):
                u_float, v_float, depth = get_pixel_from_robot_point(
                    point_3d, LEFT_T_BASE_CAMERA, CAMERA_K, CAMERA_DISTORTION
                )
                u, v = int(u_float), int(v_float)
                if depth < 0.01: continue # in a weird pose -> gripper can be behind the camera, leading to negative depth
                
                if 0 <= u < cv_image.shape[1] and 0 <= v < cv_image.shape[0]:
                    cv2.circle(cv_image, (u, v), 3, (255, 150, 150), -1)
                    # Depth 표시 (m 단위, 소수점 둘째자리)
                    text = f"{depth:.2f}m"
                    cv2.putText(cv_image, text, (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 200, 200), 1)

            # --- DRAW RIGHT ROD (RED) ---
            for i, point_3d in enumerate(self.right_rod_points):
                u_float, v_float, depth = get_pixel_from_robot_point(
                    point_3d, RIGHT_T_BASE_CAMERA, CAMERA_K, CAMERA_DISTORTION
                )
                u, v = int(u_float), int(v_float)
                if depth < 0.01: continue # in a weird pose -> gripper can be behind the camera, leading to negative depth
                
                if 0 <= u < cv_image.shape[1] and 0 <= v < cv_image.shape[0]:
                    cv2.circle(cv_image, (u, v), 3, (150, 150, 255), -1)
                    # Depth 표시
                    text = f"{depth:.2f}m"
                    cv2.putText(cv_image, text, (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 255), 1)
            
            # --- Resize and Show ---
            height, width = cv_image.shape[:2]
            display_image = cv2.resize(cv_image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)

            cv2.imshow("Robot Projection with Depth", display_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = Display3Dto2DNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()