from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'fr5_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('lib', 'python3.10', 'site-packages', 'fr5_control', 'fairino'),
         glob('fr5_control/fairino/*.so')),
        # 프로브 기하 · 안전 한계 · 루프 주기의 단일 진실 공급원 (DESIGN_NOTES §4.2)
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rosota',
    maintainer_email='rosotarun@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # --- 초음파 (RUS) ---
            'us_servo = fr5_control.us_servo_node:main',
            'us_admittance = fr5_control.us_admittance_node:main',
            'us_supervisor = fr5_control.us_supervisor_node:main',
            'us_force_search = fr5_control.us_force_search_node:main',
            'us_perception = fr5_control.us_perception_node:main',
            # --- 복강경 시절부터 남은 것. Phase 0 검증 후 legacy 로 옮긴다 ---
            'fr5_status = fr5_control.fr5_status_node:main',
            'fr5_servo_joint_control = fr5_control.fr5_servo_joint_control_node:main',
            'fr5_servo_joint_control_limit = fr5_control.fr5_servo_joint_control_node_limit:main',
            'fr5_dummy = fr5_control.fr5_dummy_test_node:main',
        ],
    },
)
