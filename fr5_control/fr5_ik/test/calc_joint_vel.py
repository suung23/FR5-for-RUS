import PyKDL as kdl
import numpy as np
import math

# --- 1. Setup Data ---
# Your defined transform from J6 to Rod Center
T_j6_rod_center_np = np.array([
    [math.cos(math.radians(25)), -math.sin(math.radians(25)), 0.0, 0.16],
    [-math.sin(math.radians(25)), -math.cos(math.radians(25)), 0.0, -0.04],
    [0.0, 0.0, -1, 0.21],
    [0.0, 0.0, 0.0, 1.0]
], dtype=float)

def numpy_to_kdl_frame(T_np):
    """Converts a 4x4 numpy matrix to a KDL Frame."""
    rot = kdl.Rotation(
        T_np[0,0], T_np[0,1], T_np[0,2],
        T_np[1,0], T_np[1,1], T_np[1,2],
        T_np[2,0], T_np[2,1], T_np[2,2]
    )
    vec = kdl.Vector(T_np[0,3], T_np[1,3], T_np[2,3])
    return kdl.Frame(rot, vec)

def build_chain_with_rod():
    """Builds the chain from Base -> J6 -> Rod Center."""
    chain = kdl.Chain()

    # --- Segments 1-6 (Same as your code) ---
    chain.addSegment(kdl.Segment(kdl.Joint("j1", kdl.Joint.RotZ),
        kdl.Frame(kdl.Rotation.RPY(1.5708, 0, 0), kdl.Vector(0, 0, 0.152))))
    
    chain.addSegment(kdl.Segment(kdl.Joint("j2", kdl.Joint.RotZ),
        kdl.Frame(kdl.Rotation.RPY(0, 0, 0), kdl.Vector(-0.425, 0, 0))))
    
    chain.addSegment(kdl.Segment(kdl.Joint("j3", kdl.Joint.RotZ),
        kdl.Frame(kdl.Rotation.RPY(0, 0, 0), kdl.Vector(-0.39501, 0, 0))))
    
    chain.addSegment(kdl.Segment(kdl.Joint("j4", kdl.Joint.RotZ),
        kdl.Frame(kdl.Rotation.RPY(1.5708, 0, 0), kdl.Vector(0, 0, 0.1021))))
    
    chain.addSegment(kdl.Segment(kdl.Joint("j5", kdl.Joint.RotZ),
        kdl.Frame(kdl.Rotation.RPY(-1.5708, 0, 0), kdl.Vector(0, 0, 0.102))))
    
    chain.addSegment(kdl.Segment(kdl.Joint("j6", kdl.Joint.RotZ),
        kdl.Frame.Identity()))

    #--- NEW SEGMENT: J6 to Rod Center ---
    #We add a "None" joint (Fixed joint) and the offset frame
    rod_frame_kdl = numpy_to_kdl_frame(T_j6_rod_center_np)
    chain.addSegment(kdl.Segment("rod", kdl.Joint("rod_joint", kdl.Joint.Fixed), rod_frame_kdl))

    return chain


def solve_dls(J, twist, damping=0.05):
    """
    Solves J * q_dot = twist using Damped Least Squares.
    Equation: q_dot = J.T * inv(J * J.T + lambda^2 * I) * twist
    """
    m, n = J.shape
    lambda_sq = damping ** 2
    identity = np.eye(m)
    
    # Calculate DLS inverse
    # This form is efficient when m <= n (rows <= cols)
    # Inverse term: (J * J.T + lambda^2 * I)^-1
    inv_term = np.linalg.inv(J @ J.T + lambda_sq * identity)
    
    # Calculate q_dot
    q_dot = J.T @ inv_term @ twist
    return q_dot



def calc_joint_velocities(current_joint_positions, wanted_twist_in_rod_frame):
    """
    Calculates needed joint velocities.
    
    Args:
        current_joint_positions: list/array of 6 joint angles [rad]
        wanted_twist_in_rod_frame: list/array of 6 values [vx, vy, vz, wx, wy, wz]
                                   Velocity relative to Rod Frame axes.
    Returns:
        np.array of 6 joint velocities [rad/s]
    """
    # 1. Initialize Solvers
    chain = build_chain_with_rod()
    jac_solver = kdl.ChainJntToJacSolver(chain)
    fk_solver = kdl.ChainFkSolverPos_recursive(chain)
    
    # 2. Prepare Joint Array
    q = kdl.JntArray(6)
    for i in range(6):
        q[i] = current_joint_positions[i]

    # 3. Calculate Jacobian at Rod Center (Expressed in Space Frame)
    # The KDL Jacobian relates joint rates -> Twist of tip (in Space Basis)
    J_kdl = kdl.Jacobian(6)
    jac_solver.JntToJac(q, J_kdl)
    
    # Convert KDL Jacobian to Numpy
    J = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            J[i, j] = J_kdl[i, j]

    # 4. Transform the Wanted Twist to Space Frame Basis
    # We need the Rotation matrix from Space to Rod to rotate the velocity vectors
    frame_rod = kdl.Frame()
    fk_solver.JntToCart(q, frame_rod)
    
    R_space_rod = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            R_space_rod[i, j] = frame_rod.M[i, j]
            
    # Split desired twist into Linear (v) and Angular (w)
    v_rod = np.array(wanted_twist_in_rod_frame[0:3])
    w_rod = np.array(wanted_twist_in_rod_frame[3:6])
    
    # Rotate them to align with Space Frame axes
    # (Note: We only rotate, we don't translate, because the "point" of reference 
    # is the rod center for both the Jacobian and this twist)
    v_space = R_space_rod @ v_rod
    w_space = R_space_rod @ w_rod
    
    wanted_twist_space = np.concatenate((v_space, w_space))

    # 5. Solve: J * q_dot = twist_space  =>  q_dot = pinv(J) * twist_space
    # Using damped least squares
    q_dot = solve_dls(J, wanted_twist_space, damping=0.05)
    
    return q_dot

# --- Usage Example ---
if __name__ == "__main__":
    # Current robot pose (Home position)
    current_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Desired Twist: Move forward along the Rod's Z-axis at 0.1 m/s
    # [vx, vy, vz, wx, wy, wz] defined in Rod Frame
    desired_twist_rod = [0.0, -5.0, 0.0, 0.0, 0.0, 0.0]
    
    q_velocities = calc_joint_velocities(current_q, desired_twist_rod)
    
    print("Desired Twist (Rod Frame):", desired_twist_rod)
    print("Calculated Joint Velocities:", np.round(q_velocities, 4))