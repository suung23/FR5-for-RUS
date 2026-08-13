from setuptools import find_packages, setup

package_name = 'fr5_ik'

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
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            #'free_control = fr5_ik.free_control_node:main', # deprecated
            #'rcm_control = fr5_ik.rcm_control_node:main',
            'rcm_two_twist = fr5_ik.rcm_two_twist:main',
            'rcm_two_pos = fr5_ik.rcm_two_pos:main',
            'freespace_two_twist = fr5_ik.freespace_two_twist:main',
            'freespace_two_pos = fr5_ik.freespace_two_pos:main',
            'rcm_two_delta = fr5_ik.rcm_two_delta:main',
            'calc_gripper_wrt_new_rcm = fr5_ik.calc_gripper_wrt_new_rcm_node:main',
            'rcm_two_delta_new_rcm = fr5_ik.rcm_two_delta_new_rcm:main',
            'rcm_two_absolute_new_rcm = fr5_ik.rcm_two_absolute_new_rcm:main'
        ],
    },
)
