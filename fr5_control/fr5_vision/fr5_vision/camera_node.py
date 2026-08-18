#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        
        # Parameters
        self.declare_parameter('device_id', 0)
        self.declare_parameter('fps', 30.0)
        
        device_id = self.get_parameter('device_id').value
        fps = self.get_parameter('fps').value
        
        # Publisher
        self.publisher_ = self.create_publisher(Image, '/camera/image1', 10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Camera Init
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            self.get_logger().error(f'Could not open video device {device_id}')
            return
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Disable Auto Exposure (1 = Manual, 3 = Auto in V4L2)
        #self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
        #self.cap.set(cv2.CAP_PROP_EXPOSURE, 100)
        
        # Timer
        timer_period = 1.0 / fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info(f'Camera Node Started: Device {device_id}, Output 640x480 RGB')

    def timer_callback(self):
        ret, frame = self.cap.read()
        
        if ret:
            # --- [추가된 부분] GUI로 화면 보여주기 ---
            # OpenCV imshow는 BGR 포맷을 사용하므로, 변환 전의 raw frame을 사용합니다.
            cv2.imshow("Endoscope Camera View", cv2.resize(frame, (1280, 960)))
            
            # 창을 갱신하기 위해 필수 (1ms 대기)
            # 여기서 'q'를 누르면 창이 꺼지는 로직 등을 추가할 수도 있습니다.
            cv2.waitKey(1)
            # -------------------------------------

            # Timestamp capture immediately
            msg_time = self.get_clock().now().to_msg()
            
            # BGR -> RGB (ROS 메시지용 변환)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to ROS Message
            try:
                msg = self.bridge.cv2_to_imgmsg(frame_rgb, encoding="rgb8")
                msg.header.stamp = msg_time
                msg.header.frame_id = "camera_optical_frame"
                
                self.publisher_.publish(msg)
            except Exception as e:
                self.get_logger().error(f'Failed to publish image: {str(e)}')
        else:
            self.get_logger().warn('Failed to capture frame')

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        # 창 닫기
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 창 확실히 닫기
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()