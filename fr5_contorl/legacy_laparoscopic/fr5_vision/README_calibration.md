# FR5 Hand-Eye Calibration Data Collection Guide

This guide describes how to collect synchronized image and robot pose data for hand-eye calibration using the `fr5_vision` package.

## Prerequisites

Ensure your workspace is built and sourced:
```bash
cd ~/FR5/ros2_ws
colcon build --symlink-install --packages-select fr5_vision fr5_control
source install/setup.bash
```

## Step-by-Step Execution

You need to open **3 separate terminals** and run the following commands in order.

### Terminal 1: Robot API & TF Broadcaster
Starts the robot connection and broadcasts TF (`fr5_right_base_link` -> `fr5_right_tool_center_point`).
```bash
source install/setup.bash
ros2 run fr5_control fr5_servo_joint_control_node
```
> **Check**: Ensure the robot connects successfully and you see `Servo Mode Started`.

### Terminal 2: Camera Node
Starts the camera stream and publishes to `/camera/image1`.
```bash
source install/setup.bash
ros2 run fr5_vision camera_node
```

### Terminal 3: Data Collector
Starts the GUI for data collection.
```bash
source install/setup.bash
ros2 run fr5_vision save_calibration_data
```

## How to Collect Data

1.  **Preparation**:
    - Print a ChArUco board (Default: 5x7 corners, Dictionary `DICT_4X4_50`).
    - Place the board in the robot's workspace so it is static.
    - Mount the camera on the robot ("Eye-in-Hand") or fix it externally ("Eye-to-Hand").

2.  **Usage**:
    - A window titled **"Calibration Data Collector"** will appear.
    - If the ChArUco board is detected, the markers will be outlined in **Green** and the text "Detected: X Cores" will appear.
    - **Move the robot** to a new pose where the board is visible.
    - Press **`s`** key on the window to save the current data sample.
        - Terminal will log: `Saved sample X to ...`

3.  **Finish**:
    - Collect 10-20 samples from different distances and angles.
    - Press **`q`** to quit.

## Data Output

Data is saved in:
`~/FR5/ros2_ws/src/fr5_vision/calibration_data`

For each sample `N`, three files are created:
- `img_N.png`: Raw image captured from camera.
- `pose_N.json`: Robot Tool Pose (Translation x,y,z + Rotation quaternion x,y,z,w).
- `charuco_N.json`: Detected marker IDs and corner coordinates (for verification).

## Configuration

To change ChArUco board settings (Dictionary, Grid Size, Square Length), edit:
`src/fr5_vision/fr5_vision/save_calibration_data.py`

Look for the configuration section at the top of the file:
```python
# [Configuration] ChArUco Board Parameters
CHARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 7
CHARUCO_SQUARE_LEN = 0.04
CHARUCO_MARKER_LEN = 0.03
```

## Step 4: Calculate Camera Intrinsics

Intrinsic calibration is required for accurate Hand-Eye calibration. The `calibrate_intrinsics` node is an interactive tool that lets you verify marker detection in real-time and capture high-quality frames.

1.  **Start Camera**: Ensure the camera node is running (as per Terminal 2).
2.  **Run Calibration Node**:
    ```bash
    ros2 run fr5_vision calibrate_intrinsics
    ```
3.  **Usage**:
    -   **Verify Detection**: A GUI window will open. Ensure the ChArUco board is detected (Green overlay).
    -   **Capture ('s')**: Move the board to different angles/distances and press **`s`** to capture. Collect 15-20 frames.
    -   **Calibrate ('c')**: Press **`c`** to calculate the intrinsics.
    -   **Output**: Results are saved to `src/fr5_vision/calibration_data/camera_intrinsics.json`.
    -   **Quit ('q')**: Press **`q`** to exit the node.

## Step 5: Calculate Hand-Eye Calibration

Once you have collected data (and optionally intrinsics), run:

```bash
ros2 run fr5_vision calculate_calibration --algorithm DANIILIDIS --intrinsics src/fr5_vision/calibration_data/camera_intrinsics.json
```

### Arguments
- `--dir`: Directory containing collected data (Default: `src/fr5_vision/calibration_data`)
- `--algorithm`: Calibration method. Options: `TSAI`, `PARK`, `HORAUD`, `ANDREFF`, `DANIILIDIS` (Default).
- `--intrinsics`: Path to `camera_intrinsics.json` generated in Step 4.

### Output
The script will print the **Camera to/in Gripper Frame** transformation matrix (Eye-in-Hand result).
You can inspect the `img_*.png` and `pose_*.json` files in the data directory if needed.

