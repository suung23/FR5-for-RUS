"""
Keyboard Input Node for Freespace Position Controller (Right Arm Only).
Generates Pose commands based on keyboard input.

Features:
1. Manual Mode: 
   - Press key -> Moves 10cm over 1.0s (Speed: 10cm/s).
   - Linear interpolation.
2. Auto Cycle Mode: 4-Phase Cycle (Total 4 seconds)
   - Move(1s) -> Hold Top(1s) -> Return(1s) -> Hold Bottom(1s)

Target Node: freespace_pos_controller_100hz.py
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import sys, select, termios, tty
import threading
import time
import copy

# Settings
MOVE_DIST = 0.10        # 10cm per move (Manual & Cycle)
MOVE_DURATION = 1.0     # 1.0s (Resulting in 10cm/s speed)
PUBLISH_RATE = 100.0    # Hz

msg = """
------------------------------------------
Fr5 Right Arm: LINEAR MOVE & CYCLE
------------------------------------------
[Manual Mode] (1 Press = 10cm Move at 10cm/s)
  w/x : X-axis (+/-)
  a/d : Y-axis (+/-)
  q/z : Z-axis (+/-)
  * Robot moves for 1s, then stops.

[Auto Cycle Mode] (Speed: 10cm/s, Period: 4s)
  1. Move Out (0->10cm) : 1s
  2. Hold Top (10cm)    : 1s
  3. Return   (10->0cm) : 1s
  4. Hold Bot (0cm)     : 1s
  
  r : Cycle X-axis
  t : Cycle Y-axis
  y : Cycle Z-axis

s : Force STOP (Stops immediately at current pos)

CTRL-C to quit
------------------------------------------
Waiting for initial pose...
"""

def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

class KeyboardCommander(Node):
    def __init__(self):
        super().__init__('keyboard_commander_right')
        
        self.pub = self.create_publisher(Pose, '/fr5_right/desired_pose_wrt_rcm', 10)
        self.sub = self.create_subscription(Pose, '/fr5_right/current_gripper_wrt_rcm', self.current_pose_cb, 10)
        
        self.current_pose = None
        self.target_pose = None
        
        # anchor_pose: 움직임의 기준점이 되는 위치 (이동 전 위치)
        self.initial_anchor_pose = None 
        self.pose_initialized = False

        # Mode State
        self.mode = "IDLE"  # IDLE, MANUAL_MOVING, CYCLE
        
        # Moving State Variables
        self.active_axis = None   # 'x', 'y', 'z'
        self.active_sign = 0      # +1, -1
        self.start_time = 0.0
        
        self.timer = self.create_timer(1.0 / PUBLISH_RATE, self.control_loop)
        
        print(msg)

    def current_pose_cb(self, msg):
        if not self.pose_initialized:
            self.current_pose = msg
            self.target_pose = copy.deepcopy(msg)
            self.initial_anchor_pose = copy.deepcopy(msg)
            self.pose_initialized = True
            print("Initial Pose Received! Ready.")

    def control_loop(self):
        if not self.pose_initialized:
            return

        if self.mode == "CYCLE":
            self.update_cycle_trajectory()
        elif self.mode == "MANUAL_MOVING":
            self.update_manual_trajectory()
        
        self.pub.publish(self.target_pose)

    def update_manual_trajectory(self):
        """
        10cm를 1초 동안 이동하는 궤적 생성 (Linear)
        """
        now = time.time()
        elapsed = now - self.start_time
        
        if elapsed < MOVE_DURATION:
            # 이동 중 (Linear Interpolation)
            ratio = elapsed / MOVE_DURATION
            offset = MOVE_DIST * ratio * self.active_sign
            
            # 기준점(Anchor)에서 offset만큼 더해서 target 계산
            self.target_pose = copy.deepcopy(self.initial_anchor_pose)
            self._apply_offset(self.target_pose, self.active_axis, offset)
            
        else:
            # 이동 완료 (정확히 10cm 지점에 도달)
            offset = MOVE_DIST * self.active_sign
            self.target_pose = copy.deepcopy(self.initial_anchor_pose)
            self._apply_offset(self.target_pose, self.active_axis, offset)
            
            # 이동이 끝났으므로 기준점(Anchor)을 현재 위치로 업데이트하고 IDLE 상태로 복귀
            self.initial_anchor_pose = copy.deepcopy(self.target_pose)
            self.mode = "IDLE"
            print("Manual Move Completed.")

    def update_cycle_trajectory(self):
        """
        4초 주기 사이클 (이동-정지-복귀-정지)
        """
        now = time.time()
        elapsed = now - self.start_time
        cycle_period = 4.0
        phase_time = elapsed % cycle_period
        
        offset = 0.0
        
        if phase_time < 1.0: # Move Out
            ratio = phase_time / 1.0
            offset = MOVE_DIST * ratio
        elif phase_time < 2.0: # Hold Top
            offset = MOVE_DIST
        elif phase_time < 3.0: # Return
            return_time = phase_time - 2.0
            ratio = 1.0 - (return_time / 1.0)
            offset = MOVE_DIST * ratio
        else: # Hold Bottom
            offset = 0.0

        self.target_pose = copy.deepcopy(self.initial_anchor_pose)
        self._apply_offset(self.target_pose, self.active_axis, offset)

    def _apply_offset(self, pose, axis, value):
        if axis == 'x': pose.position.x += value
        elif axis == 'y': pose.position.y += value
        elif axis == 'z': pose.position.z += value

    def start_manual_move(self, axis, sign):
        if not self.pose_initialized: return
        
        # 이미 움직이는 중이면 명령 무시 (안전을 위해)
        if self.mode == "MANUAL_MOVING":
            print("Already moving! Wait until finished.")
            return

        self.stop_any_motion() # Reset anchor logic
        
        self.mode = "MANUAL_MOVING"
        self.active_axis = axis
        self.active_sign = sign
        self.start_time = time.time()
        
        direction = "+" if sign > 0 else "-"
        print(f"Starting Manual Move: {axis.upper()} {direction}10cm (1.0s)...")

    def start_cycle(self, axis):
        if not self.pose_initialized: return
        self.stop_any_motion()
        
        self.mode = "CYCLE"
        self.active_axis = axis
        self.start_time = time.time()
        print(f"Started Cycle on {axis.upper()}")

    def stop_any_motion(self):
        """
        현재 동작을 멈추고 그 자리에 정지
        """
        if self.mode != "IDLE":
            # 현재 target_pose를 새로운 anchor로 삼아 정지
            self.initial_anchor_pose = copy.deepcopy(self.target_pose)
            self.mode = "IDLE"
            print("Motion Stopped. Holding Position.")

def main(args=None):
    global settings
    settings = termios.tcgetattr(sys.stdin)

    rclpy.init(args=args)
    node = KeyboardCommander()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while True:
            key = getKey()
            if key == '\x03': break
            
            if not node.pose_initialized: continue

            # Manual Move Inputs (Triggers 1s trajectory)
            if key == 'w':   node.start_manual_move('x', 1)
            elif key == 'x': node.start_manual_move('x', -1)
            elif key == 'a': node.start_manual_move('y', 1)
            elif key == 'd': node.start_manual_move('y', -1)
            elif key == 'q': node.start_manual_move('z', 1)
            elif key == 'z': node.start_manual_move('z', -1)

            # Cycle Inputs
            elif key == 'r': node.start_cycle('x')
            elif key == 't': node.start_cycle('y')
            elif key == 'y': node.start_cycle('z')
            
            # Stop
            elif key == 's': node.stop_any_motion()

    except Exception as e:
        print(e)

    finally:
        node.stop_any_motion()
        node.destroy_node()
        rclpy.shutdown()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

if __name__ == '__main__':
    main()