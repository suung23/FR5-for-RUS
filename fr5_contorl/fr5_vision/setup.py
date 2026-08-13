from setuptools import find_packages, setup

package_name = 'fr5_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'camera_node = fr5_vision.camera_node:main',
            'save_calibration_data = fr5_vision.save_calibration_data:main',
            'calculate_calibration = fr5_vision.calculate_calibration:main',
            'calibrate_intrinsics = fr5_vision.calibrate_intrinsics:main',
            'display_3d_to_2d = fr5_vision.display_3d_to_2d_node:main',
            'gt_sparse_depth = fr5_vision.gt_sparse_depth_node:main',
        ],
    },
)
