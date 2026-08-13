import numpy as np
import cv2

def get_pixel_from_robot_point(p_base, T_bc, K, D):
    """
    Maps a 3D point from Robot Base Frame to 2D Pixel coordinates.
    
    Args:
        p_base (np.array): [x, y, z] or [x, y, z, 1] in Robot Base Frame.
        T_bc (np.array): 4x4 Transformation Matrix of CAMERA pose in BASE frame.
                         (Position/Orientation of Camera relative to Robot Base).
        K (np.array): 3x3 Intrinsic Matrix.
        D (np.array): Distortion Coefficients.
    """
    
    # --- STEP 1: INVERT TRANSFORMATION ---
    # We are given T_bc (Where the camera IS).
    # We need T_cb (How to move a point INTO the camera frame).
    T_cb = np.linalg.inv(T_bc)
    
    # --- STEP 2: PREPARE DATA FOR OPENCV ---
    # Extract Rotation Matrix (3x3) and Translation Vector (3,) from the INVERTED matrix
    R_matrix = T_cb[:3, :3]
    t_vec = T_cb[:3, 3]
    
    # Convert Rotation Matrix to Rodrigues Vector (Compact rotation representation)
    r_vec, _ = cv2.Rodrigues(R_matrix)
    
    # Handle both 3-element and 4-element (homogeneous) point inputs
    if len(p_base) == 4:
        p_xyz = p_base[:3]
    else:
        p_xyz = p_base
        
    object_points = np.array([p_xyz], dtype=np.float32)

    # --- STEP 3: PROJECT POINTS ---
    # cv2.projectPoints handles the full pinhole model + distortion
    # It effectively does: s * [u,v,1] = K * (R*P + t) (with distortion)
    image_points, jacobian = cv2.projectPoints(
        object_points,
        r_vec,
        t_vec,
        K,
        D
    )
    
    # --- OUTPUT ---
    # image_points result shape is (N, 1, 2)
    uv = image_points[0][0]
    u, v = uv[0], uv[1]
    
    return u, v

# ==========================================
#        CONFIGURATION (Using T_bc)
# ==========================================

# 1. Intrinsic Matrix (K)
# Replace with your actual calibration!
fx, fy = 600.0, 600.0
cx, cy = 320.0, 240.0
K_real = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=np.float32)

# 2. Distortion Coefficients (D)
# Replace with your actual calibration!
D_real = np.array([-0.1, 0.01, 0.0, 0.0, 0.0], dtype=np.float32)

# 3. Camera Pose in Base Frame (T_bc)
# "I know my camera is at x=0.2, y=0.0, z=0.5 relative to the robot base"
T_bc_measured = np.eye(4)
T_bc_measured[:3, 3] = [0.2, 0.0, 0.5] # Example position

# 4. The Robot Point (P_base)
# A point on the tool tip
P_robot = np.array([0.15, 0.05, 0.1]) 

# ==========================================
#             EXECUTION
# ==========================================

print("--- Inputs ---")
print(f"Camera Position (Base Frame): {T_bc_measured[:3, 3]}")
print(f"Target Point (Base Frame):    {P_robot}")

# Run Calculation
u, v = get_pixel_from_robot_point(P_robot, T_bc_measured, K_real, D_real)

print("\n--- Result ---")
print(f"Projected Pixel: u={u:.2f}, v={v:.2f}")

# Check visibility (Assuming 640x480 resolution)
if 0 <= u < 640 and 0 <= v < 480:
    print(">> Status: Point is visible.")
else:
    print(">> Status: Point is OUT of view.")