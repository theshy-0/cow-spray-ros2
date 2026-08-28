import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'teat_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'models'), glob(os.path.join('models', '*.pt'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xiaoyu',
    maintainer_email='2812170944@qq.com',
    description='YOLO detection, depth projection, target TF, and debug image nodes',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'detector_node = teat_detection.detector_node:main',
            'debug_node = teat_detection.debug_node:main',
            'teat_id_node = teat_detection.teat_id_node:main',
            'handeye_marker_node = teat_detection.handeye_marker_node:main',
        ],
    },
)
