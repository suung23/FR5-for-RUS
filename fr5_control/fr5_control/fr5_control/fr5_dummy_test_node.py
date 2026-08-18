import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

class FR5DummyTestNode(Node):
    def __init__(self):
        super().__init__('fr5_dummy_test_node')
        
        # 1. Publisher
        self.publisher_ = self.create_publisher(JointState, 'fr5/joint_velocity_cmds', 10)
        
        # 2. Parameters
        self.timer_period = 0.02  # 50Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        # 3. Motion State
        self.start_time = self.get_clock().now()
        
        # Smooth Base Motion
        self.amplitude = 0.15   
        self.frequency = 0.5    
        
        # Roughness/Jitter Parameters
        self.jitter_amp = 0.00  # Intensity of the "rough" vibration
        self.jitter_freq = 50.0 # Speed of the vibration (Hz)
        
        self.get_logger().info('Dummy Test Node started. Moving with "rough" circular motion...')

    def timer_callback(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Calculate elapsed time
        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9
        
        # --- Physics Logic ---
        # 1. Smooth circular base component
        base_sin = self.amplitude * math.sin(2 * math.pi * self.frequency * elapsed)
        base_cos = self.amplitude * math.cos(2 * math.pi * self.frequency * elapsed)
        
        # 2. Add high-frequency "roughness" 
        # These also average to zero so they won't cause position drift
        noise_sin = self.jitter_amp * math.sin(2 * math.pi * self.jitter_freq * elapsed)
        noise_cos = self.jitter_amp * math.cos(2 * math.pi * self.jitter_freq * elapsed)
        
        vel_rough_j4 = base_sin + noise_sin
        vel_rough_j5 = base_cos + noise_cos
        
        # Apply to all joints for the test
        msg.velocity = [vel_rough_j4, vel_rough_j5, vel_rough_j4, 
                        vel_rough_j5, vel_rough_j4, vel_rough_j5]
        
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FR5DummyTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Send a stop command before shutting down
        stop_msg = JointState()
        stop_msg.velocity = [0.0] * 6
        node.publisher_.publish(stop_msg)
        node.get_logger().info('Emergency stop sent. Exiting...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()