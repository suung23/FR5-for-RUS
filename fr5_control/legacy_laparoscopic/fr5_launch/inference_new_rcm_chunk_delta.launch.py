import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():
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
            name='fr5_servo_joint_control',
            output='screen'
        ),



        Node(
            package='fr5_ik',
            executable='rcm_two_absolute_new_rcm',
            name='rcm_two_absolute_new_rcm',
            output='screen',
        ),
        
        Node(
            package='fr5_ik',
            executable='calc_gripper_wrt_new_rcm',
            name='calc_gripper_wrt_new_rcm',
            output='screen',
        ),

        # 4. Vision Node
        Node(
            package='fr5_vision',
            executable='camera_node',
            name='camera_node',
            output='screen'
        ),
        

        # this is the master node that controls everything
        Node(
            package='fr5_inference',
            executable='suturing_new_rcm_chunk_delta',
            name='suturing_new_rcm_chunk_delta',
            output='screen',
        ),


        # dataset collection node
        Node(
            package='dataset',
            executable='inference_h5',
            name='data_collector',
            output='screen',
            # --- [추가된 부분] 파라미터 전달 ---
            parameters=[
                {'dataset_dir': dataset_dir}
            ]
        ),
    ])