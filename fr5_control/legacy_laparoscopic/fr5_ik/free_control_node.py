import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Pose
import PyKDL as kdl
import numpy as np
import math



def numpy_to_kdl_frame(T_np):
    rot = kdl.Rotation(
        T_np[0,0], T_np[0,1], T_np[0,2],
        T_np[1,0], T_np[1,1], T_np[1,2],
        T_np[2,0], T_np[2,1], T_np[2,2]
    )
    vec = kdl.Vector(T_np[0,3], T_np[1,3], T_np[2,3])
    return kdl.Frame(rot, vec)

def build_chain_with_rod(ee_transpose_matrix: np.ndarray):
    chain = kdl.Chain()
    # Segments 1-6
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

    # Rod Segment
    rod_frame_kdl = numpy_to_kdl_frame(ee_transpose_matrix)
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
#            ROS2 NODE CLASS
# ==========================================

class RodVelocityController(Node):
    def __init__(self, robot_name: str, ee_transpose_matrix: np.ndarray):
        super().__init__(f'{robot_name}_freespace_rod_velocity_controller')
        
        self.robot_name = robot_name

        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.current_q = None 
        
        # --- SAFETY: Track last time joint state was received ---
        self.last_js_time = None
        self.js_timeout_sec = 0.2

        # --- RELATIVE POSE TRACKING ---
        self.start_rod_frame = None # Will be set on first joint callback

        # Setup KDL
        self.chain = build_chain_with_rod(ee_transpose_matrix)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)

        # Publishers / Subscribers
        self.velocity_pub = self.create_publisher(JointState, f'/{self.robot_name}/joint_velocity_cmds', 10)
        
        # New Publisher for Relative Pose
        self.relative_pose_pub = self.create_publisher(Pose, f'/{self.robot_name}/relative_ee_pose', 10)
        
        self.joint_state_sub = self.create_subscription(
            JointState,
            f'/{self.robot_name}/joint_states',
            self.joint_state_callback,
            10
        )

        self.twist_sub = self.create_subscription(
            Twist,
            f'/{self.robot_name}/desired_twist',
            self.twist_callback,
            10
        )

        self.get_logger().info(f"{self.robot_name} Rod Velocity Controller Started. Waiting for joint states...")

    def joint_state_callback(self, msg):
        """
        Updates self.current_q, updates timestamp, and publishes relative pose.
        """
        if len(msg.name) < 6:
            return

        # --- SAFETY: Update timestamp ---
        self.last_js_time = self.get_clock().now()

        pos_dict = dict(zip(msg.name, msg.position))
        try:
            ordered_q = [pos_dict[name] for name in self.joint_names]
            self.current_q = np.array(ordered_q)
            
            # --- CALCULATE AND PUBLISH RELATIVE POSE ---
            self.publish_relative_pose(self.current_q)

        except KeyError as e:
            self.get_logger().warn(f"Missing joint in joint_states: {e}")

    def publish_relative_pose(self, q_np):
        """
        Calculates FK using current joints.
        Captures start_rod_frame if not set.
        Publishes Pose relative to start frame.
        """
        # Convert numpy q to KDL JntArray
        q_kdl = kdl.JntArray(6)
        for i in range(6):
            q_kdl[i] = q_np[i]

        # Get Current Frame (Rod Center)
        current_frame = kdl.Frame()
        self.fk_solver.JntToCart(q_kdl, current_frame)

        # Initialize start frame if this is the first valid reading
        if self.start_rod_frame is None:
            self.start_rod_frame = current_frame
            self.get_logger().info(f"{self.robot_name} Initial End-Effector Frame Captured.")
        
        # Calculate Relative Frame: T_rel = T_start_inverse * T_current
        # This gives the current pose expressed in the starting frame
        relative_frame = self.start_rod_frame.Inverse() * current_frame
        
        # Create and Populate Pose Message
        pose_msg = Pose()
        
        # Position
        pose_msg.position.x = relative_frame.p.x()
        pose_msg.position.y = relative_frame.p.y()
        pose_msg.position.z = relative_frame.p.z()
        
        # Orientation (Quaternion)
        # KDL GetQuaternion returns (x, y, z, w)
        qx, qy, qz, qw = relative_frame.M.GetQuaternion()
        pose_msg.orientation.x = qx
        pose_msg.orientation.y = qy
        pose_msg.orientation.z = qz
        pose_msg.orientation.w = qw
        
        self.relative_pose_pub.publish(pose_msg)

    def twist_callback(self, msg):
        """
        Triggered when a desired twist is received.
        Checks for data staleness before publishing.
        """
        # 1. Check if we have ever received data
        if self.current_q is None or self.last_js_time is None:
            self.get_logger().warn(f"{self.robot_name} Cannot move: No Joint States received yet.", throttle_duration_sec=1.0)
            return

        # 2. --- SAFETY: Check if data is stale ---
        now = self.get_clock().now()
        time_diff = (now - self.last_js_time).nanoseconds / 1e9 
        
        if time_diff > self.js_timeout_sec:
            self.get_logger().warn(
                f"{self.robot_name} SAFETY STOP: Joint States are stale! Last received {time_diff:.3f}s ago. Not publishing velocities.",
                throttle_duration_sec=0.5
            )
            return

        # 3. Parse Twist
        wanted_twist_rod = [
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z
        ]

        # 4. Calculate and Publish
        try:
            q_dot = self.calc_joint_velocities(self.current_q, wanted_twist_rod)
            # self.get_logger().info(f"Calculated joint velocities: {q_dot}", throttle_duration_sec=1.0)
            self.publish_velocity_command(q_dot)
        except Exception as e:
            self.get_logger().error(f"Calculation error: {e}")

    def calc_joint_velocities(self, current_joint_positions, wanted_twist_in_rod_frame):
        # Setup KDL Joint Array
        q = kdl.JntArray(6)
        for i in range(6):
            q[i] = current_joint_positions[i]

        # Calculate Jacobian
        J_kdl = kdl.Jacobian(6)
        self.jac_solver.JntToJac(q, J_kdl)
        
        J = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                J[i, j] = J_kdl[i, j]

        # Calculate Rotation (Space -> Rod)
        frame_rod = kdl.Frame()
        self.fk_solver.JntToCart(q, frame_rod)
        
        R_space_rod = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                R_space_rod[i, j] = frame_rod.M[i, j]

        # Rotate Twist to Space Frame
        v_rod = np.array(wanted_twist_in_rod_frame[0:3])
        w_rod = np.array(wanted_twist_in_rod_frame[3:6])
        
        v_space = R_space_rod @ v_rod
        w_space = R_space_rod @ w_rod
        
        wanted_twist_space = np.concatenate((v_space, w_space))

        # Solve DLS
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

    # transform matrix between joint 6 to rod_center 
    left_ee_transpose_matrix = np.array([
        [math.cos(math.radians(27.3)), -math.sin(math.radians(27.3)), 0.0, 0.2475],
        [-math.sin(math.radians(27.3)), -math.cos(math.radians(27.3)), 0.0, -0.082],
        [0.0, 0.0, -1, 0.194],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)

    right_ee_transpose_matrix = np.array([
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.22],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)
    
    left_node = RodVelocityController('fr5_left', ee_transpose_matrix=left_ee_transpose_matrix)
    right_node = RodVelocityController('fr5_right', ee_transpose_matrix=right_ee_transpose_matrix)
    
    executor = MultiThreadedExecutor()
    executor.add_node(left_node)
    executor.add_node(right_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        left_node.destroy_node()
        right_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()