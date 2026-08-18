import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Pose
import PyKDL as kdl
import numpy as np
import math

# --- RCM Drift Correction Gain ---
RCM_GAIN_P = 5.0 

def numpy_to_kdl_frame(T_np):
    rot = kdl.Rotation(
        T_np[0,0], T_np[0,1], T_np[0,2],
        T_np[1,0], T_np[1,1], T_np[1,2],
        T_np[2,0], T_np[2,1], T_np[2,2]
    )
    vec = kdl.Vector(T_np[0,3], T_np[1,3], T_np[2,3])
    return kdl.Frame(rot, vec)

def build_chain_with_rod(rod_matrix):
    """
    Builds the kinematic chain using the provided rod_matrix (numpy 4x4)
    for the tool transformation.
    """
    chain = kdl.Chain()
    # Segments 1-6 (FR5 Robot DH Parameters)
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

    # Rod Segment (Applied dynamically)
    rod_frame_kdl = numpy_to_kdl_frame(rod_matrix)
    chain.addSegment(kdl.Segment("rod", kdl.Joint("rod_joint", kdl.Joint.Fixed), rod_frame_kdl))
    return chain

# solve damped least squares for safety
def solve_dls(J, twist, damping=0.05):
    m, n = J.shape
    lambda_sq = damping ** 2
    identity = np.eye(m)
    inv_term = np.linalg.inv(J @ J.T + lambda_sq * identity)
    q_dot = J.T @ inv_term @ twist
    return q_dot

# ==========================================
#             ROS2 NODE CLASS
# ==========================================

class RodVelocityController(Node):
    def __init__(self, robot_name, tool_config):
        """
        robot_name: 'fr5_left' or 'fr5_right'
        tool_config: {'x':..., 'y':..., 'z':..., 'angle_deg':...}
        """
        super().__init__(f'{robot_name}_rcm_rod_velocity_controller') 

        self.robot_name = robot_name
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.current_q = None 
        
        self.last_js_time = None
        self.js_timeout_sec = 0.5
        self.start_rod_frame = None 
        self.rcm_position_world = None 

        # --- Dynamic Tool Matrix Generation ---
        angle_rad = math.radians(tool_config['angle_deg'])
        x_off = tool_config['x']
        y_off = tool_config['y']
        z_off = tool_config['z']

        # Calculate Transform Matrix based on config
        self.T_rod_matrix = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad), 0.0, x_off],
            [-math.sin(angle_rad), -math.cos(angle_rad), 0.0, y_off],
            [0.0, 0.0, -1, z_off],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=float)

        self.get_logger().info(f"[{self.robot_name}] Tool Config: Offset({x_off}, {y_off}, {z_off}), Angle({tool_config['angle_deg']})")

        # Setup KDL (Pass the calculated matrix)
        self.chain = build_chain_with_rod(self.T_rod_matrix)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)

        # Publishers / Subscribers
        self.velocity_pub = self.create_publisher(JointState, f'/{self.robot_name}/joint_velocity_cmds', 10)
        self.relative_pose_pub = self.create_publisher(Pose, f'/{self.robot_name}/relative_ee_pose', 10)
        
        self.joint_state_sub = self.create_subscription(
            JointState, f'/{self.robot_name}/joint_states', self.joint_state_callback, 10
        )

        self.twist_sub = self.create_subscription(
            Twist, f'/{self.robot_name}/desired_twist', self.twist_callback, 10
        )

        self.get_logger().info(f"RCM Rod Controller for {self.robot_name} Started. Waiting for joint states...")

    def joint_state_callback(self, msg):
        if len(msg.name) < 6:
            return

        self.last_js_time = self.get_clock().now()

        try:
            pos_dict = dict(zip(msg.name, msg.position))
            ordered_q = [pos_dict[name] for name in self.joint_names]
            self.current_q = np.array(ordered_q)
            
            # --- CALCULATE FK & PUBLISH ---
            self.update_kinematics_and_rcm(self.current_q)

        except KeyError as e:
            # Occurs if joint_states topic doesn't match expected names
            pass

    def update_kinematics_and_rcm(self, q_np):
        # Convert numpy q to KDL JntArray
        q_kdl = kdl.JntArray(6)
        for i in range(6):
            q_kdl[i] = q_np[i]

        # Get Current Frame (Rod Center in World)
        current_frame = kdl.Frame()
        self.fk_solver.JntToCart(q_kdl, current_frame)

        # Initialize start frame & RCM point
        if self.start_rod_frame is None:
            self.start_rod_frame = current_frame
            self.rcm_position_world = current_frame.p 
            self.get_logger().info(f"RCM Point Fixed at World: {self.rcm_position_world}")
        
        # Publish Relative Pose
        relative_frame = self.start_rod_frame.Inverse() * current_frame
        pose_msg = Pose()
        pose_msg.position.x = relative_frame.p.x()
        pose_msg.position.y = relative_frame.p.y()
        pose_msg.position.z = relative_frame.p.z()
        qx, qy, qz, qw = relative_frame.M.GetQuaternion()
        pose_msg.orientation.x = qx
        pose_msg.orientation.y = qy
        pose_msg.orientation.z = qz
        pose_msg.orientation.w = qw
        self.relative_pose_pub.publish(pose_msg)

    def twist_callback(self, msg):
        # 1. Check Idle
        is_idle = (abs(msg.linear.x) < 1e-4 and abs(msg.linear.y) < 1e-4 and abs(msg.linear.z) < 1e-4 and 
                   abs(msg.angular.x) < 1e-4 and abs(msg.angular.y) < 1e-4 and abs(msg.angular.z) < 1e-4)

        # 2. Safety Checks
        if self.current_q is None or self.last_js_time is None or self.rcm_position_world is None:
            if not is_idle:
                self.get_logger().warn("Not ready: No Joint States or RCM not initialized.", throttle_duration_sec=1.0)
            return

        now = self.get_clock().now()
        time_diff = (now - self.last_js_time).nanoseconds / 1e9 
        if time_diff > self.js_timeout_sec:
            if not is_idle:
                 self.get_logger().warn(f"Joint state data is stale ({time_diff:.3f}s old). Ignoring command.", throttle_duration_sec=1.0)
            return 

        # [CORE LOGIC]
        q_kdl = kdl.JntArray(6)
        for i in range(6): q_kdl[i] = self.current_q[i]
        
        current_rod_frame_world = kdl.Frame()
        self.fk_solver.JntToCart(q_kdl, current_rod_frame_world)

        rcm_in_rod_frame = current_rod_frame_world.Inverse() * self.rcm_position_world

        L = rcm_in_rod_frame.x()
        y_err = rcm_in_rod_frame.y()
        z_err = rcm_in_rod_frame.z()

        v_ux = msg.linear.x
        w_ux = msg.angular.x
        w_uy = msg.angular.y
        w_uz = msg.angular.z

        v_y_corrected = (-L * w_uz) + (RCM_GAIN_P * y_err)
        v_z_corrected = (L * w_uy) + (RCM_GAIN_P * z_err)

        final_twist_rod = [v_ux, v_y_corrected, v_z_corrected, w_ux, w_uy, w_uz]

        try:
            q_dot = self.calc_joint_velocities(self.current_q, final_twist_rod)
            self.publish_velocity_command(q_dot)
        except Exception as e:
            self.get_logger().error(f"Calculation error: {e}")

    def calc_joint_velocities(self, current_joint_positions, wanted_twist_in_rod_frame):
        q = kdl.JntArray(6)
        for i in range(6): q[i] = current_joint_positions[i]

        J_kdl = kdl.Jacobian(6)
        self.jac_solver.JntToJac(q, J_kdl)
        
        J = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                J[i, j] = J_kdl[i, j]

        frame_rod = kdl.Frame()
        self.fk_solver.JntToCart(q, frame_rod)
        
        R_space_rod = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                R_space_rod[i, j] = frame_rod.M[i, j]

        v_rod = np.array(wanted_twist_in_rod_frame[0:3])
        w_rod = np.array(wanted_twist_in_rod_frame[3:6])
        
        v_space = R_space_rod @ v_rod
        w_space = R_space_rod @ w_rod
        
        wanted_twist_space = np.concatenate((v_space, w_space))

        return solve_dls(J, wanted_twist_space, damping=0.05)

    def publish_velocity_command(self, velocities):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names 
        msg.velocity = velocities.tolist()
        msg.position = []
        msg.effort = []
        self.velocity_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    
    # TOOL CONFIGURATION
    
    # Left Arm
    left_config = {
        'x': 0.36,         # Stick Length
        'y': -0.1,         # Lateral Offset
        'z': 0.21,         # Vertical Offset
        'angle_deg': 27.3  # Angle
    }

    # Right Arm
    right_config = {
        'x': 0.30,       
        'y': -0.205,        
        'z': 0.21,        
        'angle_deg': 46.0 
    }
    
    left_node = RodVelocityController('fr5_left', left_config)
    right_node = RodVelocityController('fr5_right', right_config)
    
    executor = MultiThreadedExecutor()
    executor.add_node(left_node)
    executor.add_node(right_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if executor:
            executor.shutdown()
        try: 
            left_node.destroy_node()
            right_node.destroy_node()
        except:
            pass

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()