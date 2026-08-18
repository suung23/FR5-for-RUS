import h5py
import numpy as np
import cv2
import os
import glob
import PyKDL as kdl
import math
from tqdm import tqdm

# =========================================================
# 1. 상수 및 설정 (gt_sparse_depth_node.py 값 적용)
# =========================================================

# Transform Matrices
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

# Camera Parameters
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

# Base to Camera Transforms
LEFT_T_BASE_CAMERA = np.array([
    [-0.7183621723048073, 0.6810479177084175, -0.14187855083326228, -0.7834750631033105],
    [0.6954791330915809, 0.6983031249766463, -0.16935619587730896, -0.3081507504430141],
    [-0.01626544913923518, -0.2203326563015092, -0.9752891651871354, 0.2378386148606645],
    [0.0, 0.0, 0.0, 1.0]
], dtype=float)

RIGHT_T_BASE_CAMERA = np.array([
    [-0.669916120911863, 0.7279718780917964, -0.14584010302349393, -0.38433242202041085],
    [0.7410206602573638, 0.6434843839078406, -0.19187555534378786, -0.7098847537661764],
    [-0.045834179540394766, -0.23661105716819147, -0.9705227640872768, 0.2363934720545219],
    [0.0, 0.0, 0.0, 1.0]
], dtype=float)

ROD_LENGTH = 0.1 
ROD_POINTS_COUNT = 20 

# =========================================================
# 2. 로봇 기구학 및 투영 함수
# =========================================================

def numpy_to_kdl_frame(mat):
    rot = kdl.Rotation(mat[0,0], mat[0,1], mat[0,2], mat[1,0], mat[1,1], mat[1,2], mat[2,0], mat[2,1], mat[2,2])
    vec = kdl.Vector(mat[0,3], mat[1,3], mat[2,3])
    return kdl.Frame(rot, vec)

def build_fr5_chain():
    chain = kdl.Chain()
    chain.addSegment(kdl.Segment("j1", kdl.Joint("j1", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(1.5708, 0, 0), kdl.Vector(0, 0, 0.152))))
    chain.addSegment(kdl.Segment("j2", kdl.Joint("j2", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(0, 0, 0), kdl.Vector(-0.425, 0, 0))))
    chain.addSegment(kdl.Segment("j3", kdl.Joint("j3", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(0, 0, 0), kdl.Vector(-0.39501, 0, 0))))
    chain.addSegment(kdl.Segment("j4", kdl.Joint("j4", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(1.5708, 0, 0), kdl.Vector(0, 0, 0.1021))))
    chain.addSegment(kdl.Segment("j5", kdl.Joint("j5", kdl.Joint.RotZ), kdl.Frame(kdl.Rotation.RPY(-1.5708, 0, 0), kdl.Vector(0, 0, 0.102))))
    chain.addSegment(kdl.Segment("j6", kdl.Joint("j6", kdl.Joint.RotZ), kdl.Frame.Identity()))
    return chain

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

def calculate_rod_points(fk_solver, q, gripper_transform):
    frame_j6 = kdl.Frame()
    fk_solver.JntToCart(q, frame_j6)
    gripper_wrt_base = frame_j6 * gripper_transform 
    points_3d = []
    for i in range(ROD_POINTS_COUNT):
        dist = (ROD_LENGTH / (ROD_POINTS_COUNT - 1)) * i
        point_wrt_base = gripper_wrt_base * kdl.Vector(-dist, 0, 0)
        points_3d.append([point_wrt_base.x(), point_wrt_base.y(), point_wrt_base.z()])
    return points_3d

# =========================================================
# 3. Vision Snapping 함수 (핵심 보정 로직)
# =========================================================

def snap_points_to_rod_vision(cv_image, projected_points, search_radius=50):
    if not projected_points:
        return []

    # 1. FK 좌표들만 추출 (u, v)
    pts_uv = np.array([[p[0], p[1]] for p in projected_points], dtype=np.int32)
    
    # 2. 마스크 생성 (ROI 설정)
    mask_roi = np.zeros(cv_image.shape[:2], dtype=np.uint8)
    cv2.polylines(mask_roi, [pts_uv], isClosed=False, color=255, thickness=search_radius*2)
    
    # 3. 검은색상 추출 (HSV 활용)
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60]) 
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # 4. ROI와 검은색 마스크 교집합
    final_mask = cv2.bitwise_and(mask_black, mask_black, mask=mask_roi)
    
    # 5. 검은색 픽셀들의 좌표 추출
    y_idxs, x_idxs = np.where(final_mask > 0)
    
    # 감지된 픽셀이 너무 적으면 보정하지 않고 원본 FK 사용
    if len(x_idxs) < 50: 
        return projected_points

    # 6. 직선 피팅 (Line Fitting)
    points_for_fitting = np.column_stack((x_idxs, y_idxs)).astype(np.float32)
    [vx, vy, x0, y0] = cv2.fitLine(points_for_fitting, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # 7. 점 투영 (Snapping)
    corrected_points = []
    vec_line = np.array([vx[0], vy[0]])
    point_on_line = np.array([x0[0], y0[0]])
    
    for pt in projected_points:
        u_fk, v_fk, depth = pt
        p_curr = np.array([u_fk, v_fk])
        
        vec_ap = p_curr - point_on_line
        dot_prod = np.dot(vec_ap, vec_line)
        p_proj = point_on_line + dot_prod * vec_line
        
        corrected_points.append([p_proj[0], p_proj[1], depth])
        
    return corrected_points

# =========================================================
# 4. H5 파일 처리 로직 (Group '0', '1', '2'...)
# =========================================================

def process_h5_file(filepath, chain, fk_solver, left_grip_kdl, right_grip_kdl):
    print(f"Processing: {filepath}")
    
    try:
        # r+ 모드: 읽기 및 덮어쓰기 허용
        with h5py.File(filepath, 'r+') as f:
            
            # 파일 내 모든 키 확인
            keys = list(f.keys())
            
            # 숫자로 된 키(Step)만 골라서 숫자 기준으로 정렬 (0, 1, 2, ..., 10, 11)
            # 문자열 정렬 시 '10'이 '2'보다 앞에 오므로 int 변환 후 정렬 필수
            step_keys = sorted([k for k in keys if k.isdigit()], key=lambda x: int(x))
            
            if not step_keys:
                print(f"Skipping {filepath}: No numerical step groups found.")
                return

            print(f" -> Found {len(step_keys)} steps. Fixing...")
            
            for step_key in tqdm(step_keys, desc="Steps", leave=False):
                grp = f[step_key]
                
                # --- 1. 데이터 로드 ---
                # 이미지 (Compressed)
                if 'image1' not in grp: continue
                
                img_data = grp['image1'][()]
                if img_data.size == 0: continue
                
                # 디코딩 (jpg bytes -> numpy bgr)
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                if img is None: continue

                # Joint States
                left_j = grp['fr5_left_joint_states'][()]
                right_j = grp['fr5_right_joint_states'][()]
                
                # --- 2. FK & Rod Point 생성 ---
                # Left Arm
                q_l = kdl.JntArray(6)
                for i in range(6): q_l[i] = left_j[i]
                left_rod_pts_3d = calculate_rod_points(fk_solver, q_l, left_grip_kdl)
                
                # Right Arm
                q_r = kdl.JntArray(6)
                for i in range(6): q_r[i] = right_j[i]
                right_rod_pts_3d = calculate_rod_points(fk_solver, q_r, right_grip_kdl)
                
                # --- 3. Projection (3D -> 2D pixel) ---
                projected_left = []
                for p3 in left_rod_pts_3d:
                    u, v, d = get_pixel_from_robot_point(p3, LEFT_T_BASE_CAMERA, CAMERA_K, CAMERA_DISTORTION)
                    # 화면 밖이나 Depth 이상한 값 필터링
                    if 0.01 < d < 100 and 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                        projected_left.append([u, v, d])
                        
                projected_right = []
                for p3 in right_rod_pts_3d:
                    u, v, d = get_pixel_from_robot_point(p3, RIGHT_T_BASE_CAMERA, CAMERA_K, CAMERA_DISTORTION)
                    if 0.01 < d < 100 and 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                        projected_right.append([u, v, d])
                        
                # --- 4. Vision Snapping (보정) ---
                # 여기서 중복 없이, 직선에 정렬된 점들만 반환됨
                snapped_left = snap_points_to_rod_vision(img, projected_left)
                snapped_right = snap_points_to_rod_vision(img, projected_right)
                
                # --- 5. 데이터 병합 ---
                final_points = snapped_left + snapped_right
                
                if len(final_points) > 0:
                    new_gt = np.array(final_points, dtype=np.float32)
                else:
                    new_gt = np.empty((0, 3), dtype=np.float32)
                    
                # --- 6. 데이터 덮어쓰기 ---
                if 'gt_sparse_depths' in grp:
                    del grp['gt_sparse_depths'] # 기존 데이터 삭제
                
                # 새 데이터 생성 (압축 없이 저장하거나 필요시 chunks 옵션 사용)
                grp.create_dataset('gt_sparse_depths', data=new_gt)
                
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

# =========================================================
# 5. 실행부
# =========================================================
if __name__ == "__main__":
    # KDL Setup
    chain = build_fr5_chain()
    fk_solver = kdl.ChainFkSolverPos_recursive(chain)
    left_grip_kdl = numpy_to_kdl_frame(FR5_LEFT_GRIPPER_WRT_J6_TRANS)
    right_grip_kdl = numpy_to_kdl_frame(FR5_RIGHT_GRIPPER_WRT_J6_TRANS)
    
    # 경로 설정 (중요: 본인의 데이터 경로로 수정)
    data_dir = "/home/rosota/FR5/ros2_ws/collected_data/suture_annotated/" 
    
    # H5 파일 검색
    h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    print(f"Found {len(h5_files)} H5 files.")
    
    for f in h5_files:
        process_h5_file(f, chain, fk_solver, left_grip_kdl, right_grip_kdl)
        
    print("All datasets have been fixed.")