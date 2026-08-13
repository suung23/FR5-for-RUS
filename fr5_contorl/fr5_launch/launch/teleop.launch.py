import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():
    # 0. Argument 설정
    
    # IK 모드 설정 (기본값: free)
    ik_mode_arg = DeclareLaunchArgument(
        'ik_mode',
        default_value='free',
        description='Select IK mode: "free" or "rcm"'
    )

    # --- [추가된 부분] 데이터 저장 경로 설정 ---
    dataset_dir_arg = DeclareLaunchArgument(
        'dataset_dir',
        default_value='collected_data',
        description='Path to save dataset files'
    )

    ik_mode = LaunchConfiguration('ik_mode')
    dataset_dir = LaunchConfiguration('dataset_dir')

    return LaunchDescription([
        ik_mode_arg,
        dataset_dir_arg, # 추가

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

        # 3-A. IK Node (Free Mode)
        Node(
            package='fr5_ik',
            executable='freespace_two_twist',
            name='freespace_two_twist',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", ik_mode, "' == 'free'"])
            )
        ),

        # 3-B. IK Node (RCM Mode)
        Node(
            package='fr5_ik',
            executable='rcm_two_twist',
            name='rcm_two_twist',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", ik_mode, "' == 'rcm'"])
            )
        ),

        # 4. Vision Node
        Node(
            package='fr5_vision',
            executable='camera_node',
            name='camera_node',
            output='screen'
        ),
        
        # 4.5. Vision - Sparse depth calculation node


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