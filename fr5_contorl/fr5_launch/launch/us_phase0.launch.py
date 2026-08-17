"""Phase 0 골격: 서보 + 미분 IK (DESIGN_NOTES §15).

    ros2 launch fr5_launch us_phase0.launch.py                # mock, 하드웨어 불필요
    ros2 launch fr5_launch us_phase0.launch.py backend:=fairino

두 노드 모두 fr5_control/config/probe.yaml 을 읽는다. 값은 거기서 고친다.
``backend`` 인자만 명령줄에서 덮어쓰는데, 실로봇과 mock 을 오가는 일이 잦고
그때마다 파일을 고치면 실수로 커밋될 수 있기 때문이다.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("fr5_control"), "config", "probe.yaml"
    )

    backend_arg = DeclareLaunchArgument(
        "backend",
        default_value="mock",
        description='로봇 백엔드: "mock" (하드웨어 없음) 또는 "fairino" (실로봇)',
    )
    teleop_arg = DeclareLaunchArgument(
        "teleop",
        default_value="false",
        description="Touch 원격조작 노드를 함께 띄운다 (햅틱 장치 필요)",
    )
    backend = LaunchConfiguration("backend")

    return LaunchDescription([
        backend_arg,
        teleop_arg,

        Node(
            package="fr5_control",
            executable="us_servo",
            name="us_servo_node",
            output="screen",
            parameters=[config, {"robot.backend": backend}],
            emulate_tty=True,
        ),

        Node(
            package="fr5_ik",
            executable="us_diff_ik",
            name="us_diff_ik_node",
            output="screen",
            parameters=[config],
            emulate_tty=True,
        ),

        # Touch 원격조작. probe.yaml 의 teleop 프로파일을 읽는다.
        # 조작 중에도 프로파일을 바꿀 수 있다:
        #   ros2 param set /touch_teleop_node teleop.profile laparoscopic
        Node(
            package="touch_teleop",
            executable="touch_twist",
            name="touch_teleop_node",
            output="screen",
            parameters=[config],
            condition=IfCondition(LaunchConfiguration("teleop")),
            emulate_tty=True,
        ),
    ])
