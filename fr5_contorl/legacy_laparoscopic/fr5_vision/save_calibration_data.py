#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import os
import json
import sys

# TF2 imports
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# ==============================================================================
# [Configuration] ChArUco Board Parameters (Hardcoded for easy modification)
# ==============================================================================
# Dictionary options: cv2.aruco.DICT_4X4_50, cv2.aruco.DICT_5X5_100, etc.
CHARUCO_DICT_ID = cv2.aruco.DICT_4X4_100
CHARUCO_SQUARES_X = 12     # Number of squares in X direction
CHARUCO_SQUARES_Y = 9     # Number of squares in Y direction
CHARUCO_SQUARE_LEN = 0.008 # Square side length (in meters)
CHARUCO_MARKER_LEN = 0.006 # Marker side length (in meters)
# Robot Frame Settings
BASE_FRAME = 'fr5_right_base_link'
TOOL_FRAME = 'fr5_right_tool_center_point'
CAMERA_TOPIC = '/camera/image1'
SAVE_DIR = '/home/rosota/FR5/ros2_ws/src/fr5_vision/calibration_data_right'
# ==============================================================================

class CalibrationCollectorNode(Node):
    def __init__(self):
        super().__init__('calibration_collector_node')
        
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

        # 2. Setup TF Buffer & Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 3. Setup Subscriber
        self.subscription = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.image_callback,
            10
        )
        self.bridge = CvBridge()

        # 4. Setup Save Directory
        self.save_path = os.path.join(os.getcwd(), SAVE_DIR)
        os.makedirs(self.save_path, exist_ok=True)
        self.get_logger().info(f"Saving data to: {self.save_path}")

        # State
        self.sample_count = 0
        self.latest_frame = None
        self.image_size = None

        self.get_logger().info("Ready. Focus on the OpenCV window and press 's' to save data.")
        self.get_logger().info(f"Tracking TF: {BASE_FRAME} -> {TOOL_FRAME}")

    def image_callback(self, msg):
        try:
            # ROS Image -> OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_frame = cv_image.copy()

            # ChArUco Detection
            # 1) Detect Board
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            if self.image_size is None:
                self.image_size = gray.shape[::-1]

            charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)
            
            display_image = cv_image.copy()
            
            # Info text
            text_status = "No Board"
            color_status = (0, 0, 255) # Red

            current_charuco_corners = None
            current_charuco_ids = None

            detected = False

            # Visualization
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(display_image, marker_corners, marker_ids)
                
            if charuco_corners is not None and len(charuco_corners) > 4:
                # self.get_logger().info(f"Detected {len(charuco_corners)} charuco corners.")
                cv2.aruco.drawDetectedCornersCharuco(display_image, charuco_corners, charuco_ids, (0, 255, 0))
                text_status = f"Ready (Corners: {len(charuco_corners)})"
                color_status = (0, 255, 0) # Green
                detected = True
                
                current_charuco_corners = charuco_corners
                current_charuco_ids = charuco_ids

            elif marker_ids is not None:
                text_status = "Markers detected but not enough corners"
                color_status = (0, 0, 255) # Red

            # Overlay GUI
            cv2.putText(display_image, text_status, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_status, 2)
            cv2.putText(display_image, "Press 's' to Save", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Calibration Data Collector", display_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.save_data(self.latest_frame, current_charuco_corners, current_charuco_ids)
            elif key == ord('q'):
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")

    def save_data(self, image, corners, ids):
        # 1. Get Robot Pose (TF)
        try:
            # Look up the latest transform available
            # We use Time(seconds=0) to get the very latest, 
            # but ideally we should sync with image timestamp if possible.
            # For static gathering, latest is usually fine.
            trans = self.tf_buffer.lookup_transform(
                BASE_FRAME, 
                TOOL_FRAME, 
                rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as ex:
            self.get_logger().error(f"Could not get transform: {ex}")
            return

        # 2. Prepare Data
        # Pose
        pose_data = {
            "translation": {
                "x": trans.transform.translation.x,
                "y": trans.transform.translation.y,
                "z": trans.transform.translation.z
            },
            "rotation": {
                "x": trans.transform.rotation.x,
                "y": trans.transform.rotation.y,
                "z": trans.transform.rotation.z,
                "w": trans.transform.rotation.w
            },
            "timestamp": trans.header.stamp.sec + trans.header.stamp.nanosec * 1e-9
        }

        # ChArUco
        charuco_data = {
            "detected": False,
            "count": 0,
            "ids": [],
            "corners": [] # list of [x, y]
        }
        
        if corners is not None and ids is not None:
            charuco_data["detected"] = True
            charuco_data["count"] = len(ids)
            charuco_data["ids"] = ids.flatten().tolist()
            # corners shape is (N, 1, 2) -> flatten to (N, 2)
            charuco_data["corners"] = corners.reshape(-1, 2).tolist()

        # 3. Save to Disk
        idx = self.sample_count
        
        # Paths
        img_name = f"img_{idx}.png"
        pose_name = f"pose_{idx}.json"
        charuco_name = f"charuco_{idx}.json"
        
        img_path = os.path.join(self.save_path, img_name)
        pose_path = os.path.join(self.save_path, pose_name)
        charuco_path = os.path.join(self.save_path, charuco_name)

        # Write
        cv2.imwrite(img_path, image)
        
        with open(pose_path, 'w') as f:
            json.dump(pose_data, f, indent=4)
            
        with open(charuco_path, 'w') as f:
            json.dump(charuco_data, f, indent=4)

        self.get_logger().info(f"Saved sample {idx} to {self.save_path}")
        self.sample_count += 1


def main(args=None):
    rclpy.init(args=args)
    node = CalibrationCollectorNode()
    
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
