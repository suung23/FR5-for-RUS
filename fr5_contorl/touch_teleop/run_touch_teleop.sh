#!/bin/bash

# load workspace environment
source /home/rosota/FR5/ros2_ws/install/setup.bash

# kill existing nodes to ensure clean start
echo "Cleaning up existing nodes..."
pkill -f fr5_servo_joint_control
pkill -f rcm_control
pkill -f touch_twist
# Also kill the specific python script names just in case they are running directly
pkill -f fr5_servo_joint_control_node.py
pkill -f rcm_control_node.py

sleep 1

echo "Starting FR5 Teleoperation System..."

# 1. Run Servo Control Node (Robot Control)
echo "1. Starting FR5 Servo Joint Control Node..."
ros2 run fr5_control fr5_servo_joint_control &
SERVO_PID=$!
# Wait for robot connection & gripper test (approx 10s)
sleep 15

# 2. Run RCM Control Node (IK / Kinematics)
echo "2. Starting RCM Control Node (fr5_ik/rcm_control)..."
ros2 run fr5_ik rcm_control &
RCM_PID=$!
sleep 2

# 3. Run Touch Twist Node (Haptic Teleop)
echo "3. Starting Touch Twist Node..."
ros2 run touch_teleop touch_twist &
TELEOP_PID=$!

echo "---------------------------------------"
echo "Touch Teleop System Running"
echo "Nodes:"
echo "  - Servo Control PID: $SERVO_PID"
echo "  - RCM Control PID:   $RCM_PID"
echo "  - Teleop PID:        $TELEOP_PID"
echo "---------------------------------------"
echo "Press ENTER to exit."

# Wait for user input to keep script running
read -r

# Cleanup on exit
echo "Stopping nodes..."
kill $TELEOP_PID
kill $RCM_PID
kill $SERVO_PID

# Ensure they are dead
sleep 1
pkill -f fr5_servo_joint_control
pkill -f rcm_control
pkill -f touch_twist

echo "Done."