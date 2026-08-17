from setuptools import find_packages, setup

package_name = 'fr5_inference'

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
    maintainer_email='rosota@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'suturing_delta = fr5_inference.suturing_master_delta_node:main',
            'suturing_delta_afterimage = fr5_inference.suturing_master_delta_afterimage_node:main',
            'suturing_delta_new_rcm = fr5_inference.suturing_master_delta_new_rcm_node:main',
            'suturing_absolute_new_rcm = fr5_inference.suturing_master_absolute_new_rcm_node:main',
            'suturing_new_rcm_chunk_delta = fr5_inference.suturing_master_new_rcm_chunk_delta:main'
        ],
    },
)
