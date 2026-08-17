import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension # 데이터 발행용 메시지 추가
import PyKDL as kdl
import numpy as np
import math
from cv_bridge import CvBridge
import cv2

# ==========================================
#              CONSTANTS
# ==========================================
# (기존 상수 설정 동일하게 유지)
FR5_LEFT_GRIPPER_WRT_J6_TRANS = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.615],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

FR5_RIGHT_GRIPPER_WRT_J6_TRANS = np.array([
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.615],
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
            -0.7183621723048073,
            0.6810479177084175,
            -0.14187855083326228,
            -0.7834750631033105
        ],
        [
            0.6954791330915809,
            0.6983031249766463,
            -0.16935619587730896,
            -0.3081507504430141
        ],
        [
            -0.01626544913923518,
            -0.2203326563015092,
            -0.9752891651871354,
            0.2378386148606645
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
            -0.669916120911863,
            0.7279718780917964,
            -0.14584010302349393,
            -0.38433242202041085
        ],
        [
            0.7410206602573638,
            0.6434843839078406,
            -0.19187555534378786,
            -0.7098847537661764
        ],
        [
            -0.045834179540394766,
            -0.23661105716819147,
            -0.9705227640872768,
            0.2363934720545219
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ], dtype=float)

ROD_LENGTH = 0.1 
ROD_POINTS_COUNT = 20 
CAMERA_TOPIC = '/camera/image1'

def numpy_to_kdl_frame(mat):
    rot = kdl.Rotation(mat[0,0], mat[0,1], mat[0,2], mat[1,0], mat[1,1], mat[1,2], mat[2,0], mat[2,1], mat[2,2])
    vec = kdl.Vector(mat[0,3], mat[1,3], mat[2,3])
    return kdl.Frame(rot, vec)


def get_pixel_from_robot_point(p_base, T_bc, K, D):
    T_cb = np.linalg.inv(T_bc)
    p_base_hom = np.array([p_base[0], p_base[1], p_base[2], 1.0])
    p_cam = T_cb @ p_base_hom
    depth = p_cam[2]
    
    R_matrix = T_cb[:3, :3]
    t_vec = T_cb[:3, 3]
    r_vec, _ = cv2.Rodrigues(R_matrix)
    object_points = np.array([p_base[:3]], dtype=np.float32)
    image_points, jacobian = cv2.projectPoints(object_points, r_vec, t_vec, K, D)
    uv = image_points[0][0]
    return uv[0], uv[1], depth




def snap_points_to_rod_vision(cv_image, projected_points, search_radius=50):
    """
    FK로 계산된 점들(projected_points)을 사용하여, 
    이미지(cv_image) 내의 실제 '검은색 막대'를 찾아 점들을 그 위로 이동시킵니다.
    
    :param cv_image: 현재 카메라 프레임 (BGR)
    :param projected_points: FK로 계산된 픽셀 좌표 리스트 [[u, v, depth], ...]
    :param search_radius: FK 좌표 주변 몇 픽셀을 검색할지 (ROI 크기)
    :return: 보정된 좌표 리스트 [[new_u, new_v, depth], ...]
    """
    if not projected_points:
        return []

    # 1. FK 좌표들만 추출 (u, v)
    pts_uv = np.array([[p[0], p[1]] for p in projected_points], dtype=np.int32)
    
    # 2. 마스크 생성 (ROI 설정)
    # 전체 이미지가 아니라, FK 점들이 있는 곳 주변만 하얀색인 마스크를 만듭니다.
    mask_roi = np.zeros(cv_image.shape[:2], dtype=np.uint8)
    # 점들을 잇는 두꺼운 선을 그려서 검색 영역(ROI)을 만듦
    cv2.polylines(mask_roi, [pts_uv], isClosed=False, color=255, thickness=search_radius*2)
    cv2.imshow("Mask roi", mask_roi)
    
    # 3. 검은색상 추출 (HSV 활용)
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    # 검은색 범위 설정 (조명에 따라 튜닝 필요할 수 있음)
    # Saturation(채도)이 낮고 Value(명도)가 낮은 것이 검은색
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60]) # 명도(V) 60 이하를 검은색으로 간주
    
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # 4. ROI와 검은색 마스크 교집합 (로봇 근처의 검은색만 남김)
    final_mask = cv2.bitwise_and(mask_black, mask_black, mask=mask_roi)
    
    # (디버깅용) 마스크 확인이 필요하면 주석 해제
    cv2.imshow("Final Mask", final_mask)

    # 5. 검은색 픽셀들의 좌표 추출
    # final_mask에서 흰색(255)인 픽셀들의 위치를 다 가져옴
    y_idxs, x_idxs = np.where(final_mask > 0)
    
    if len(x_idxs) < 50: # 감지된 픽셀이 너무 적으면 보정 포기 (그냥 원래 FK 값 리턴)
        return projected_points

    # 6. 직선 피팅 (Line Fitting)
    # 감지된 점들의 분포를 가장 잘 표현하는 직선 벡터를 구함
    points_for_fitting = np.column_stack((x_idxs, y_idxs)).astype(np.float32)
    # vx, vy: 직선의 방향 벡터 (단위 벡터)
    # x0, y0: 직선 위의 한 점 (보통 데이터의 중심)
    [vx, vy, x0, y0] = cv2.fitLine(points_for_fitting, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # 7. 점 투영 (Snapping)
    # FK 점(P)을 찾아낸 직선 위로 수직 이동시킴
    corrected_points = []
    
    # 방향 벡터 정규화
    vec_line = np.array([vx[0], vy[0]])
    point_on_line = np.array([x0[0], y0[0]])
    
    for pt in projected_points:
        u_fk, v_fk, depth = pt
        p_curr = np.array([u_fk, v_fk])
        
        # 벡터 투영 공식:
        # 직선 위의 점 A(point_on_line)에서 현재 점 P(p_curr)로 가는 벡터 AP
        vec_ap = p_curr - point_on_line
        
        # AP를 직선 방향(vec_line)에 내적 -> 직선상의 거리
        dot_prod = np.dot(vec_ap, vec_line)
        
        # 투영된 점의 위치 = A + (내적값 * 방향벡터)
        p_proj = point_on_line + dot_prod * vec_line
        
        corrected_points.append([p_proj[0], p_proj[1], depth])
        
    return corrected_points



class ProjectAndDepthNode(Node):
    def __init__(self):
        super().__init__('calc_gt_sparse_depth_node')
        
        # --- Kinematics ---
        self.chain = self.build_fr5_chain()
        self.left_gripper_frame_kdl = numpy_to_kdl_frame(FR5_LEFT_GRIPPER_WRT_J6_TRANS)
        self.right_gripper_frame_kdl = numpy_to_kdl_frame(FR5_RIGHT_GRIPPER_WRT_J6_TRANS)
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        
        # --- State ---
        self.left_q = None
        self.right_q = None
        self.left_rod_points = []
        self.right_rod_points = []

        # --- Subscribers ---
        self.sub_left_joints = self.create_subscription(
            JointState, '/fr5_left/joint_states', self.left_joint_cb, 10)
        self.sub_right_joints = self.create_subscription(
            JointState, '/fr5_right/joint_states', self.right_joint_cb, 10) 
        self.sub_image = self.create_subscription(
            Image, CAMERA_TOPIC, self.image_callback, 10)

        # --- Publishers (Depth Data) ---
        self.pub_sparse_depth = self.create_publisher(Float32MultiArray, 'gt_sparse_depth', 10)




        # for printing joint states
        self.left_j6_deg = 0.0
        self.right_j6_deg = 0.0

        # --- [추가 2] JointState 구독자 설정 ---
        # 큐 사이즈는 최신값만 필요하므로 1~10 정도면 충분합니다.
        self.create_subscription(
            JointState, 
            '/fr5_left/joint_states', 
            self.left_joint_callback, 
            10
        )
        
        self.create_subscription(
            JointState, 
            '/fr5_right/joint_states', 
            self.right_joint_callback, 
            10
        )
        self.bridge = CvBridge()
        self.get_logger().info("calc_gt_sparse_depth_node Initialized.")
        
        
    def left_joint_callback(self, msg):
        # 메시지에 최소 6개의 조인트 정보가 있는지 확인
        if len(msg.position) >= 6:
            # msg.position은 라디안(radian)이므로 도(degree)로 변환
            # 인덱스 5가 6번째 관절 (Joint 6)
            self.left_j6_deg = math.degrees(msg.position[5])

    def right_joint_callback(self, msg):
        if len(msg.position) >= 6:
            self.right_j6_deg = math.degrees(msg.position[5])
            
            
    def draw_joint_info(self, image):
        """
        이미지 왼쪽 위에 조인트 정보를 덮어씁니다.
        image: cv2 이미지 배열 (numpy array)
        """
        # 텍스트 설정
        text_left = f"Left J6:  {self.left_j6_deg:.2f} deg"
        text_right = f"Right J6: {self.right_j6_deg:.2f} deg"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 255, 0) # 초록색 (BGR)
        thickness = 2
        
        # 가독성을 위해 검은색 테두리(outline)를 먼저 그림
        cv2.putText(image, text_left, (20, 40), font, font_scale, (0, 0, 0), thickness + 3)
        cv2.putText(image, text_right, (20, 80), font, font_scale, (0, 0, 0), thickness + 3)

        # 그 위에 실제 글씨를 씀
        cv2.putText(image, text_left, (20, 40), font, font_scale, color, thickness)
        cv2.putText(image, text_right, (20, 80), font, font_scale, color, thickness)
        
        return image

    def build_fr5_chain(self):
        chain = kdl.Chain()
        chain.addSegment(kdl.Segment("j1", kdl.Joint("j1", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(1.5708, 0, 0), kdl.Vector(0, 0, 0.152))))
        chain.addSegment(kdl.Segment("j2", kdl.Joint("j2", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(0, 0, 0), kdl.Vector(-0.425, 0, 0))))
        chain.addSegment(kdl.Segment("j3", kdl.Joint("j3", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(0, 0, 0), kdl.Vector(-0.39501, 0, 0))))
        chain.addSegment(kdl.Segment("j4", kdl.Joint("j4", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(1.5708, 0, 0), kdl.Vector(0, 0, 0.1021))))
        chain.addSegment(kdl.Segment("j5", kdl.Joint("j5", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(-1.5708, 0, 0), kdl.Vector(0, 0, 0.102))))
        chain.addSegment(kdl.Segment("j6", kdl.Joint("j6", kdl.Joint.RotZ), kdl.Frame.Identity()))
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
        # 1. Check if we have joint data
        if not self.left_rod_points or not self.right_rod_points:
            return 
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            sparse_data_list = [] # [u, v, depth] 저장 리스트\
                
            # --- PROCESS LEFT ---
            left_temp_points = []
            for point_3d in self.left_rod_points:
                u_float, v_float, depth = get_pixel_from_robot_point(point_3d, LEFT_T_BASE_CAMERA, CAMERA_K, CAMERA_DISTORTION)
                if depth > 0.01 and depth < 100:
                    if 0 <= u_float < cv_image.shape[1] and 0 <= v_float < cv_image.shape[0]:
                        left_temp_points.append([u_float, v_float, depth])
            corrected_left_points = snap_points_to_rod_vision(cv_image, left_temp_points)

            # --- PROCESS RIGHT ---
            right_temp_points = []
            for point_3d in self.right_rod_points:
                u_float, v_float, depth = get_pixel_from_robot_point(point_3d, RIGHT_T_BASE_CAMERA, CAMERA_K, CAMERA_DISTORTION)
                if depth > 0.01 and depth < 100:
                    if 0 <= u_float < cv_image.shape[1] and 0 <= v_float < cv_image.shape[0]:
                        right_temp_points.append([u_float, v_float, depth])
            corrected_right_points = snap_points_to_rod_vision(cv_image, right_temp_points)
            

            # for debug
            debug_image = cv_image.copy()
            for p in left_temp_points:
                u, v, d = int(p[0]), int(p[1]), p[2]
                # Visualization
                cv2.circle(debug_image, (u, v), 3, (255, 100, 100), -1)
                cv2.putText(debug_image, f"{d:.2f}m", (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
            for p in right_temp_points:
                u, v, d = int(p[0]), int(p[1]), p[2]
                # Visualization
                cv2.circle(debug_image, (u, v), 3, (100, 100, 255), -1)
                cv2.putText(debug_image, f"{d:.2f}m", (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
            height, width = debug_image.shape[:2]
            cv2.imshow("Debug without snapping view", debug_image)
            

            # STORE and VISUALIZE
            for p in corrected_left_points:
                u, v, d = int(p[0]), int(p[1]), p[2]
                sparse_data_list.append([float(p[0]), float(p[1]), d])
                # Visualization
                cv2.circle(cv_image, (u, v), 3, (255, 100, 100), -1)
                cv2.putText(cv_image, f"{d:.2f}m", (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
            for p in corrected_right_points:
                u, v, d = int(p[0]), int(p[1]), p[2]
                sparse_data_list.append([float(p[0]), float(p[1]), d])
                # Visualization
                cv2.circle(cv_image, (u, v), 3, (100, 100, 255), -1)
                cv2.putText(cv_image, f"{d:.2f}m", (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
            
            # 2. Publish Sparse Depth Data
            array_msg = Float32MultiArray()
            
            if sparse_data_list:
                # 데이터가 있는 경우: 평탄화(Flatten) 및 차원 정보 설정
                flat_data = [element for row in sparse_data_list for element in row]
                array_msg.data = flat_data
                
                rows = len(sparse_data_list)
                cols = 3
                
                dim_rows = MultiArrayDimension(label="points", size=rows, stride=rows*cols)
                dim_cols = MultiArrayDimension(label="coords", size=cols, stride=cols)
                array_msg.layout.dim = [dim_rows, dim_cols]
            else:
                # 데이터가 없는 경우: 빈 리스트 설정
                array_msg.data = []
                array_msg.layout.dim = []

            # 조건 없이 항상 발행
            self.pub_sparse_depth.publish(array_msg)

            # 3. Visualization (Resized)
            height, width = cv_image.shape[:2]
            display_image = cv2.resize(cv_image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
            
            # display joint state
            display_image = self.draw_joint_info(display_image)
            cv2.imshow("Projected Depth View", display_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ProjectAndDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()