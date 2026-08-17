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
            # RCM(트로카 구속) 계열은 초음파에 대응물이 없어 legacy_laparoscopic/ 으로 분리됨.
            # 표면 접촉 정렬은 RCM이 아니라 Mx/My 기반 자세 정렬로 처리한다 (DESIGN_NOTES §5).
            'us_diff_ik = fr5_ik.us_diff_ik_node:main',
            'freespace_two_twist = fr5_ik.freespace_two_twist:main',
        ],
    },
)
