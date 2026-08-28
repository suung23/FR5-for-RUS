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
            # PX6D F/T 센서 시리얼 점검. 제어 경로가 아니라 배선·축 확인용이다.
            'px6d_probe = fr5_control.px6d_probe:main',
            # 같은 시리얼 경로를 실시간 그래프로. 축 배정 실측에 쓴다.
            'px6d_viz = fr5_control.px6d_viz:main',
            # 같은 시리얼 경로를 터미널에서. 화면이 없는 콘솔·SSH 용이다.
            'px6d_monitor = fr5_control.px6d_monitor:main',
            # 축 배정·부호를 손으로 눌러 확정하는 절차. AXIS_ORDER 와 §4.4 검증용.
            'px6d_axis_id = fr5_control.px6d_axis_id:main',
            'probe_tcp_id = fr5_control.probe_tcp_id:main',
            'px6d_verify = fr5_control.px6d_verify:main',
            # teleop 콘솔 GUI 가 붙는 웹소켓 브리지. 구독 전용이다.
            'telemetry_bridge = fr5_control.telemetry_bridge:main',
            # --- 복강경 시절부터 남은 것. Phase 0 검증 후 legacy 로 옮긴다 ---
            'fr5_status = fr5_control.fr5_status_node:main',
            'fr5_servo_joint_control = fr5_control.fr5_servo_joint_control_node:main',
            'fr5_servo_joint_control_limit = fr5_control.fr5_servo_joint_control_node_limit:main',
            'fr5_dummy = fr5_control.fr5_dummy_test_node:main',
        ],
    },
)
