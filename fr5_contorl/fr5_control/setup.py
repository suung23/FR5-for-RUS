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
            'fairino_driver = fr5_control.fairino_driver:main',
            'fr5_status = fr5_control.fr5_status_node:main',
            'fr5_servo_joint_control = fr5_control.fr5_servo_joint_control_node:main',
            'fr5_servo_joint_control_limit = fr5_control.fr5_servo_joint_control_node_limit:main',
            'fr5_dummy = fr5_control.fr5_dummy_test_node:main',
            'fr5_dual_direct = fr5_control.fr5_dual_direct_control_node:main',
            'fr5_dual_direct_joint = fr5_control.fr5_dual_direct_joint_control_node:main',


        ],
    },
)
