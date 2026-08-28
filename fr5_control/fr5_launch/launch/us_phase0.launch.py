"""Phase 0 골격: 서보 + 미분 IK (DESIGN_NOTES §15).

    ros2 launch fr5_launch us_phase0.launch.py                       # mock, 하드웨어 불필요
    ros2 launch fr5_launch us_phase0.launch.py backend:=fairino      # 실로봇
    ros2 launch fr5_launch us_phase0.launch.py teleop:=true          # freespace 스케일 (기본)
    ros2 launch fr5_launch us_phase0.launch.py teleop:=true freespace:=false   # 접촉용 상한

세 노드 모두 fr5_control/config/probe.yaml 을 읽는다. 값은 거기서 고친다.
명령줄 인자는 셋뿐이며, 모두 "실로봇과 mock 을 오가는" 류의 잦은 전환용이다.
파일을 고치면 실수로 커밋되기 때문이다.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# freespace 스케일. 2026-08-25 부터 **기본값**이고, freespace:=false 로 끈다.
#
# probe.yaml 의 safety 값은 **접촉 중** 기준이다 (§12.3). teleop 으로 초기 자세에
# 접근하는 동안에는 그 값이 지나치게 답답해서, 실제 운용은 거의 항상 이쪽을 켠 채
# 이루어졌다. 기본값이 실사용과 어긋나 있으면 매번 인자를 붙여야 하고, 붙이는 것을
# 잊은 run 이 "로봇이 안 움직인다" 로 오진된다 (2026-08-19 가 그 사례다).
#
# probe.yaml 을 직접 고치지 않는 이유는 그대로다 — 접촉용 값이 어딘가 한 곳에
# 남아 있어야 되돌릴 수 있다. 여기는 그 값을 덮어쓰는 자리이지 값의 출처가 아니다.
#
# ⚠️ 대신 위험의 방향이 뒤집혔다. 프로브가 조직·팬텀에 닿는 단계에서는
# **freespace:=false 를 명시**해야 한다. 잊으면 접촉용 상한이 아니라 150 mm/s 로 돈다.
#
# teleop 프로파일도 같이 바꾼다. 프로파일만 바꾸면 클램프에 걸려 체감이 안 변하고,
# 클램프만 올리면 스케일이 낮아 여전히 느리다. 둘은 함께 움직여야 한다.
FREESPACE_OVERRIDES = {
    "teleop.profile": "freespace",
    # 스케일만 올리면 이 클램프에 걸려 체감이 안 바뀐다. 함께 올린다.
    # max_joint_vel 1.5 는 DESIGN_NOTES §12.3 이 "자유공간 기준"으로 언급한 값이다.
    #
    # 2026-08-19: 스케일과 함께 0.6 / 3.0 / 3.0 까지 올렸다가 되돌렸다. 실제로
    # 조작해 보니 너무 빨랐다. 상향이 필요해 보였던 것은 us_diff_ik 가 죽어 있어
    # 로봇이 안 움직인 탓이었고, 속도 한계 탓이 아니었다.
    #
    # 실질적 천장은 관절 상한이다 — 카테시안 상한을 올려도 IK 출력은 서보 단에서
    # 다시 잘린다. 특이점 근처에서 DLS 가 큰 관절속도를 내므로 이 값이 곧
    # 최악의 경우 속도이며, 그래서 필요 이상으로 올려두지 않는다.
    "safety.max_linear_vel_m_s": 0.15,      # 10 → 150 mm/s
    # 2026-08-27: 1.5 → 0.9 로 내렸다. 툴 변환(당시 243 mm)이 들어오면서 회전이
    # 프로브 끝을 중심으로 돌게 됐고, 같은 각속도에 필요한 관절속도가 늘었다.
    # 뻗은 자세에서 1.2 rad/s 지령이 관절 2.04 rad/s 를 요구해 아래
    # max_joint_vel(1.5)를 넘겼다 — 넘으면 서보가 그 관절만 잘라내고, 잘린
    # 관절 하나가 **전체 운동 방향을 튼다.** 조작자에게는 "회전이 말을 안 듣고
    # 자세마다 다르다" 로 느껴진다.
    #
    # 0.9 는 최악 자세에서도 관절 요구를 1.5 아래로 두는 값이다. 회전이 느려지는
    # 대신 **지령한 방향으로 간다.** 병진은 그대로다 (툴 길이와 무관하다).
    #
    # 같은 날 늦게 툴이 234 mm 로 줄었다 (마운트를 CAD 로 대체). 필요한 관절속도가
    # 4 % 줄었을 뿐이라 이 값은 그대로 두고, 보수적인 쪽으로 남긴다.
    "safety.max_angular_vel_rad_s": 0.9,    # 0.2 → 0.9 rad/s
    "safety.max_joint_vel_rad_s": 1.5,      # 0.5 → 1.5 rad/s
}


def _launch_setup(context, *args, **kwargs):
    config = os.path.join(
        get_package_share_directory("fr5_control"), "config", "probe.yaml"
    )
    backend = LaunchConfiguration("backend").perform(context)
    staging = LaunchConfiguration("contact_probing").perform(context).lower() not in (
        "false", "0", "no"
    )
    freespace = LaunchConfiguration("freespace").perform(context).lower() in ("true", "1")

    # 노드마다 **사본**을 넘긴다. 같은 dict 객체를 세 노드에 공유하면 launch_ros 가
    # 정규화하면서 그것을 소비해, 리스트에서 먼저 생성되는 노드에만 적용되고
    # 나머지는 조용히 원래값으로 돈다. 실제로 us_servo 만 먹고 us_diff_ik 는
    # 접촉용 상한(0.010 m/s)을 그대로 쓰고 있었다 — 눈에 띄지 않는 종류의 버그다.
    def _params(*extra):
        out = [config]
        out.extend(e for e in extra if e)
        if freespace:
            out.append(dict(FREESPACE_OVERRIDES))
        return out

    servo_params = _params({"robot.backend": backend})
    ik_params = _params(None if staging else {"teleop.contact_probing_enabled": False})
    teleop_params = _params()

    return [
        Node(
            package="fr5_control",
            executable="us_servo",
            name="us_servo_node",
            output="screen",
            parameters=servo_params,
            emulate_tty=True,
        ),
        Node(
            package="fr5_ik",
            executable="us_diff_ik",
            name="us_diff_ik_node",
            output="screen",
            parameters=ik_params,
            emulate_tty=True,
        ),
        # Touch 원격조작.
        #   버튼 1 (회색): 누르고 있는 동안에만 동작한다 (데드맨). 놓으면 정지.
        #   프로파일은 조작 중에도 바꿀 수 있다:
        #     ros2 param set /touch_teleop_node teleop.profile us_approach
        Node(
            package="touch_teleop",
            executable="touch_twist",
            name="touch_teleop_node",
            output="screen",
            parameters=teleop_params,
            condition=IfCondition(LaunchConfiguration("teleop")),
            emulate_tty=True,
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "contact_probing",
            default_value="true",
            description=(
                "F_n 문턱에서 접촉 프로빙으로 전환할지. false 면 힘이 얼마가 되든 "
                "접근 상한을 유지한다 — 전자저울 검증처럼 의도적으로 문턱을 넘겨 "
                "누르면서 teleop 을 계속해야 하는 절차 전용이다. 전환은 단방향이라 "
                "한 번 걸리면 그 세션에서 다시 못 나온다. ⚠️ 안전 거동을 끄는 것이며, "
                "남는 층은 힘 한계뿐이다"
            ),
        ),
        DeclareLaunchArgument(
            "backend",
            default_value="mock",
            description='로봇 백엔드: "mock" (하드웨어 없음) 또는 "fairino" (실로봇)',
        ),
        DeclareLaunchArgument(
            "teleop",
            default_value="false",
            description="Touch 원격조작 노드를 함께 띄운다 (햅틱 장치 필요)",
        ),
        DeclareLaunchArgument(
            "freespace",
            default_value="true",
            description=(
                "freespace 스케일 (기본). teleop 프로파일을 freespace 로 두고 "
                "속도 상한을 올린다. 프로브가 조직·팬텀에 닿는 단계에서는 "
                "false 로 내려 probe.yaml 의 접촉용 한계(§12.3)를 쓸 것"
            ),
        ),
        OpaqueFunction(function=_launch_setup),
    ])
