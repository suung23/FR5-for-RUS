from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 0. Argument 설정
    dataset_dir_arg = DeclareLaunchArgument(
        'dataset_dir',
        default_value='collected_data',
        description='Path to save dataset files'
    )

    dataset_dir = LaunchConfiguration('dataset_dir')

    return LaunchDescription([
        dataset_dir_arg,

        # 1. Control Node
        Node(
            package='fr5_control',
            executable='fr5_servo_joint_control_limit',
            name='fr5_servo_joint_control_limit',
            output='screen'
        ),

        # 2. Touch Node
        Node(
            package='touch_teleop',
            executable='touch_twist',
            name='touch_teleop_node',
            output='log',
        ),

        # 3. IK Node
        # RCM 모드는 legacy_laparoscopic/ 으로 분리됨 — 초음파는 트로카 구속이 없다.
        Node(
            package='fr5_ik',
            executable='freespace_two_twist',
            name='freespace_two_twist',
            output='screen',
        ),

        # 4. Vision Node
        Node(
            package='fr5_vision',
            executable='camera_node',
            name='camera_node',
            output='screen'
        ),

        # 5. Data Collector Node
        Node(
            package='dataset',
            executable='fr5_h5',
            name='data_collector',
            output='screen',
            # --- [추가된 부분] 파라미터 전달 ---
            parameters=[
                {'dataset_dir': dataset_dir}
            ]
        ),
    ])