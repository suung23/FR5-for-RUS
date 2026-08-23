#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import json
import argparse
from datetime import datetime

# ==============================================================================
# [Configuration] ChArUco Board Parameters
# ==============================================================================
CHARUCO_DICT_ID = cv2.aruco.DICT_4X4_100
CHARUCO_SQUARES_X = 12     # Number of squares in X direction
CHARUCO_SQUARES_Y = 9     # Number of squares in Y direction
CHARUCO_SQUARE_LEN = 0.008 # Square side length (in meters)
CHARUCO_MARKER_LEN = 0.006 # Marker side length (in meters)
CAMERA_TOPIC = '/camera/image1'
DEFAULT_SAVE_DIR = '/home/rosota/ALES/ros2_ws/src/fr5_vision/new_camera_calibration_data'
# ==============================================================================

class IntrinsicCalibrationNode(Node):
    def __init__(self, save_dir):
        super().__init__('intrinsic_calibration_node')
        
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # 1. Setup ChArUco
        self.dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT_ID)
        self.board = cv2.aruco.CharucoBoard(
            (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
            CHARUCO_SQUARE_LEN,
            CHARUCO_MARKER_LEN,
            self.dictionary
        )
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board, detectorParams=self.detector_params)

        # 2. Setup Subscriber
        self.subscription = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        
        # 3. Storage for Calibration
        self.all_corners = [] # List of charuco_corners
        self.all_ids = []     # List of charuco_ids
        self.image_size = None
        
        self.get_logger().info(f"Intrinsic Calibration Node Started.")
        self.get_logger().info(f"Subscribed to {CAMERA_TOPIC}")
        self.get_logger().info(f"Press 's' to capture a frame. Press 'c' to calibrate. Press 'q' to quit.")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            if self.image_size is None:
                self.image_size = gray.shape[::-1]

            # Detect Board (replacing verify markers + interpolate)
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)
            
            display_image = cv_image.copy()
            status_text = "Press 's' to Capture | 'c' to Calibrate"
            status_color = (0, 255, 255) # Yellow
            
            detected = False
            #self.get_logger().info(f"Marker IDs detected: {len(marker_ids.flatten())if marker_ids is not None else 'None'}")

            # Visualization
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(display_image, marker_corners, marker_ids)
                
            if charuco_corners is not None and len(charuco_corners) > 4:
                self.get_logger().info(f"Detected {len(charuco_corners)} charuco corners.")
                cv2.aruco.drawDetectedCornersCharuco(display_image, charuco_corners, charuco_ids, (0, 255, 0))
                status_text = f"Ready (Corners: {len(charuco_corners)})"
                status_color = (0, 255, 0) # Green
                detected = True
            elif marker_ids is not None:
                if charuco_corners is not None:
                    self.get_logger().info(f"Detected {len(charuco_corners)} charuco corners.")

                status_text = "Markers detected but not enough corners"
                status_color = (0, 0, 255) # Red

            # Overlay Text
            cv2.putText(display_image, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display_image, f"Captured Frames: {len(self.all_corners)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Intrinsic Calibration", display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                if detected:
                    self.capture_frame(cv_image, charuco_corners, charuco_ids)
                else:
                    self.get_logger().warn("Cannot capture: Not enough charuco corners detected.")
            elif key == ord('c'):
                self.run_calibration()
            elif key == ord('q'):
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")

    def capture_frame(self, image, charuco_corners, charuco_ids):
        self.all_corners.append(charuco_corners)
        self.all_ids.append(charuco_ids)
        
        # Save image for record
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"calib_img_{timestamp}_{len(self.all_corners)}.png"
        cv2.imwrite(os.path.join(self.save_dir, img_filename), image)
        
        self.get_logger().info(f"Captured frame {len(self.all_corners)}. Image saved to {img_filename}")

    def run_calibration(self):
        if len(self.all_corners) < 3:
            self.get_logger().error("Not enough frames captured. Need at least 3.")
            return

        self.get_logger().info(f"Starting calibration with {len(self.all_corners)} frames...")
        
        try:
            all_obj_points = []
            all_img_points = []
            
            # Match object points
            for i in range(len(self.all_corners)):
                obj_pts, img_pts = self.board.matchImagePoints(self.all_corners[i], self.all_ids[i])
                all_obj_points.append(obj_pts)
                all_img_points.append(img_pts)
            
            # Calibrate Camera
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                all_obj_points, all_img_points, self.image_size, None, None
            )

            self.get_logger().info("Calibration Info:")
            self.get_logger().info(f"Reprojection Error: {ret}")
            self.get_logger().info(f"Camera Matrix:\n{mtx}")
            self.get_logger().info(f"Distortion Coeffs:\n{dist}")

            # Save results
            output_file = os.path.join(self.save_dir, 'camera_intrinsics.json')
            data = {
                "reprojection_error": ret,
                "camera_matrix": mtx.tolist(),
                "dist_coeffs": dist.tolist(),
                "image_width": self.image_size[0],
                "image_height": self.image_size[1]
            }

            with open(output_file, 'w') as f:
                json.dump(data, f, indent=4)
            
            self.get_logger().info(f"Calibration successful! Results saved to {output_file}")
            
        except Exception as e:
            self.get_logger().error(f"Calibration failed: {e}")
            import traceback
            traceback.print_exc()

def main(args=None):
    rclpy.init(args=args)
    
    # Simple arg parsing for output dir
    parser = argparse.ArgumentParser(description='Interactive Intrinsic Calibration Node')
    parser.add_argument('--dir', type=str, default=DEFAULT_SAVE_DIR, help='Directory to save captured images and results')
    
    # We need to filter out ROS args if any are passed
    ros_args, unknown_args = parser.parse_known_args()
    
    node = IntrinsicCalibrationNode(ros_args.dir)
    
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
