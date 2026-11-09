import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'safety_estop'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 문턱·복귀 속도의 단일 진실 공급원. probe.yaml 과 **따로** 둔다 —
        # 안전 문턱이 제어 설정과 같은 파일에 있으면, 제어를 고치다가 문턱이
        # 함께 움직인다.
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rosota',
    maintainer_email='rosotarun@gmail.com',
    description='비상정지·보호정지 감시와 초기 자세 복귀',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'safety_estop = safety_estop.estop_node:main',
        ],
    },
)
