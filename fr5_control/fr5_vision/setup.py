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
            # eye-to-hand 내시경 캘리브레이션 계열은 legacy_laparoscopic/ 으로 분리됨.
            # 초음파는 probe-to-image 캘리브레이션(N-wire/cross-wire 팬텀)으로 절차가 다르다.
            'camera_node = fr5_vision.camera_node:main',
            # US 프레임그래버 (DESIGN_NOTES §3). 장비 AP TCP 5002/5003 -> /us/image
            'us_frame = fr5_vision.us_frame_node:main',
            # 장비 없이 수신 경로를 돌려보기 위한 테스트 더블. 합성 영상이다.
            'us_mock_scanner = fr5_vision.us_mock_scanner:main',
        ],
    },
)
