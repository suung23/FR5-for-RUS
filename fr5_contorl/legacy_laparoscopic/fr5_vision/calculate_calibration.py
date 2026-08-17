#!/usr/bin/env python3
import cv2
import numpy as np
import json
import os
import argparse
import glob
from scipy.spatial.transform import Rotation as R

# ==============================================================================
# [Configuration] ChArUco Board Parameters
# ==============================================================================
CHARUCO_DICT_ID = cv2.aruco.DICT_4X4_100
CHARUCO_SQUARES_X = 12     # Number of squares in X direction
CHARUCO_SQUARES_Y = 9     # Number of squares in Y direction
CHARUCO_SQUARE_LEN = 0.008 # Square side length (in meters)
CHARUCO_MARKER_LEN = 0.006 # Marker side length (in meters)
# ==============================================================================

def get_calibration_method(method_name):
    methods = {
        'TSAI': cv2.CALIB_HAND_EYE_TSAI,
        'PARK': cv2.CALIB_HAND_EYE_PARK,
        'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
        'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
        'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    return methods.get(method_name.upper(), cv2.CALIB_HAND_EYE_DANIILIDIS)

def main():
    parser = argparse.ArgumentParser(description='Calculate Eye-to-Hand Calibration (Camera Fixed, Board on Robot).')
    parser.add_argument('--dir', type=str, default='/home/rosota/FR5/ros2_ws/src/fr5_vision/calibration_data_right', help='Directory containing collected data')
    parser.add_argument('--algorithm', type=str, default='DANIILIDIS', help='Calibration algorithm')
    parser.add_argument('--intrinsics', type=str, default='/home/rosota/FR5/ros2_ws/src/fr5_vision/calibration_data/camera_intrinsics.json', help='Path to camera intrinsics JSON')
    args = parser.parse_args()

    data_dir = args.dir
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist.")
        return

    # 1. Setup ChArUco
    dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_LEN,
        CHARUCO_MARKER_LEN,
        dictionary
    )
    params = cv2.aruco.DetectorParameters()

    # 2. Lists for calibration
    # Eye-to-Hand Trick: We need Gripper -> Base, not Base -> Gripper
    R_gripper2base_inv = [] 
    t_gripper2base_inv = []
    
    # Target -> Camera (Standard PnP)
    R_target2cam = []
    t_target2cam = []

    # 3. Process Files
    pose_files = glob.glob(os.path.join(data_dir, 'pose_*.json'))
    pose_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

    print(f"Found {len(pose_files)} samples. Mode: Eye-to-Hand (Fixed Camera)")

    valid_samples = 0

    for pose_path in pose_files:
        idx = os.path.basename(pose_path).split('_')[1].split('.')[0]
        img_path = os.path.join(data_dir, f'img_{idx}.png')
        charuco_json_path = os.path.join(data_dir, f'charuco_{idx}.json')

        # --- A. Process ChArUco (Target -> Camera) ---
        charuco_corners = None
        charuco_ids = None
        
        # Load Charuco Data
        if os.path.exists(charuco_json_path):
            with open(charuco_json_path, 'r') as f:
                c_data = json.load(f)
            if c_data.get('detected', False) and c_data['count'] > 0:
                charuco_ids = np.array(c_data['ids'], dtype=np.int32).reshape(-1, 1)
                charuco_corners = np.array(c_data['corners'], dtype=np.float32).reshape(-1, 1, 2)
        
        # Fallback to image detection if needed
        elif os.path.exists(img_path):
            img = cv2.imread(img_path)
            corners, ids, _ = cv2.aruco.detectMarkers(img, dictionary, parameters=params)
            if len(ids) > 0:
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, img, board)

        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 4:
            # Load Intrinsics
            if args.intrinsics and os.path.exists(args.intrinsics):
                with open(args.intrinsics, 'r') as f:
                    intr = json.load(f)
                camera_matrix = np.array(intr['camera_matrix'])
                dist_coeffs = np.array(intr['dist_coeffs'])
            else:
                print("Intrinsics needed for accurate PnP.")
                return

            # Solve PnP (Target -> Camera)
            all_obj_points = np.array(board.getChessboardCorners(), dtype=np.float32).reshape(-1, 3)
            obj_points = all_obj_points[charuco_ids.flatten(), :]
            
            valid, rvec, tvec = cv2.solvePnP(obj_points, charuco_corners, camera_matrix, dist_coeffs)
            
            if valid:
                R_t2c, _ = cv2.Rodrigues(rvec)
                t_t2c = tvec

                # --- B. Process Pose (Base -> Gripper -> INVERT -> Gripper -> Base) ---
                with open(pose_path, 'r') as f:
                    pose_data = json.load(f)
                
                # 1. Construct Transformation Matrix (Base -> Gripper)
                t_b2g = np.array([pose_data['translation']['x'], pose_data['translation']['y'], pose_data['translation']['z']])
                quat = [pose_data['rotation']['x'], pose_data['rotation']['y'], pose_data['rotation']['z'], pose_data['rotation']['w']]
                R_b2g = R.from_quat(quat).as_matrix()
                
                T_b2g = np.eye(4)
                T_b2g[:3, :3] = R_b2g
                T_b2g[:3, 3] = t_b2g
                
                # 2. Invert it! (Gripper -> Base)
                # This is the KEY Step for Eye-to-Hand using standard calibrateHandEye
                T_g2b = np.linalg.inv(T_b2g)
                
                R_g2b = T_g2b[:3, :3]
                t_g2b = T_g2b[:3, 3]

                # Append to lists
                R_gripper2base_inv.append(R_g2b)
                t_gripper2base_inv.append(t_g2b)

                R_target2cam.append(R_t2c)
                t_target2cam.append(t_t2c)
                
                valid_samples += 1
            else:
                 print(f"Sample {idx}: PnP Failed.")
        else:
            print(f"Sample {idx}: Markers not detected.")

    print(f"Used {valid_samples} valid samples for Eye-to-Hand calibration.")

    if valid_samples < 3:
        print("Error: Not enough samples.")
        return

    # 4. Run Calibration
    # Because we inverted the robot pose input, the output R, t represents BASE -> CAMERA
    method = get_calibration_method(args.algorithm)
    
    R_b2c, t_b2c = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base_inv,
        t_gripper2base=t_gripper2base_inv,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=method
    )

    # Convert to standard forms
    H_b2c = np.eye(4)
    H_b2c[:3, :3] = R_b2c
    H_b2c[:3, 3] = t_b2c.flatten()
    quat_b2c = R.from_matrix(R_b2c).as_quat()

    # 5. Output Results
    print("\n" + "="*50)
    print(f"Eye-to-Hand Calibration Result (Base -> Camera)")
    print("="*50)
    print("Translation (x, y, z) [meters]:")
    print(t_b2c.ravel())
    print("\nRotation Matrix:")
    print(R_b2c)
    print("\nHomogeneous Matrix (T_base_cam):")
    print(H_b2c)
    print("="*50)

    # 6. Save
    result_path = os.path.join(data_dir, "eye_to_hand_result.json")
    result_data = {
        "mode": "eye_to_hand",
        "description": "Transform from Robot Base to Fixed Camera (T_base_cam)",
        "valid_samples": valid_samples,
        "translation": {"x": float(t_b2c[0]), "y": float(t_b2c[1]), "z": float(t_b2c[2])},
        "quaternion_xyzw": quat_b2c.tolist(),
        "rotation_matrix": R_b2c.tolist(),
        "homogeneous_matrix": H_b2c.tolist()
    }

    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=4)
        print(f"Saved to {result_path}")

if __name__ == '__main__':
    main()