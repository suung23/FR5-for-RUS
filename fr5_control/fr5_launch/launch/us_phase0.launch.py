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
    admittance_arg = DeclareLaunchArgument(
        "admittance",
        default_value="false",
        description="admittance 노드를 띄운다. teleop 과 함께 켜도 된다 — 조작자 twist 는 "
                    "admittance 의 TELEOP 모드를 거쳐 나가므로 다투지 않는다",
    )
    control_arg = DeclareLaunchArgument(
        "control",
        default_value="false",
        description="supervisor + force_search 를 띄운다 (Stage 1/2 자동 제어)",
    )
    perception_arg = DeclareLaunchArgument(
        "perception",
        default_value="false",
        description=(
            "us_frame + us_perception 을 띄운다. frame.source 를 녹화 파일 경로로 두면 "
            "하드웨어 없이 Stage 1 전 구간을 돌릴 수 있다"
        ),
    )
    backend = LaunchConfiguration("backend")

    return LaunchDescription([
        backend_arg,
        teleop_arg,
        admittance_arg,
        control_arg,
        perception_arg,

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

        # Admittance. 힘 축(z, rx, ry)과 영상 축(x, y, rz)을 합쳐 twist 를 낸다.
        # CAD 미도착 상태에서는 tool.allow_missing_tool 때문에 항등변환으로 돌며,
        # 그때 M_x/M_y 는 지렛대 항에 오염되어 있으므로 z축 힘 추종만 유효하다.
        Node(
            package="fr5_control",
            executable="us_admittance",
            name="us_admittance_node",
            output="screen",
            parameters=[config],
            condition=IfCondition(LaunchConfiguration("admittance")),
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

        # 감독기 — 상태 전이와 모드 발행. admittance 가 이 모드를 따른다.
        Node(
            package="fr5_control",
            executable="us_supervisor",
            name="us_supervisor_node",
            output="screen",
            parameters=[config],
            condition=IfCondition(LaunchConfiguration("control")),
            emulate_tty=True,
        ),

        # Stage 1 힘 탐색. STAGE1A/1B 상태에서만 setpoint 를 낸다.
        Node(
            package="fr5_control",
            executable="us_force_search",
            name="us_force_search_node",
            output="screen",
            parameters=[config],
            condition=IfCondition(LaunchConfiguration("control")),
            emulate_tty=True,
        ),

        # 프레임 취득. frame.source 가 숫자면 장치, 아니면 녹화 파일.
        Node(
            package="fr5_vision",
            executable="us_frame",
            name="us_frame_node",
            output="screen",
            parameters=[config],
            condition=IfCondition(LaunchConfiguration("perception")),
            emulate_tty=True,
        ),

        # 지각. rus_perception 이 설치돼 있어야 한다 (pip install -e Unet_seg).
        # perception.checkpoint 가 비어 있으면 기동을 거부한다.
        Node(
            package="fr5_control",
            executable="us_perception",
            name="us_perception_node",
            output="screen",
            parameters=[config],
            condition=IfCondition(LaunchConfiguration("perception")),
            emulate_tty=True,
        ),
    ])
