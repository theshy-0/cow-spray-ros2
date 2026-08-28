import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'camera'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dev',
    maintainer_email='dev@example.com',
    description='SICK ToF camera ROS2 driver node.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_node = camera.camera_node:main',
            'camera_config = camera.camera_config:main',
            'image_collect_node = camera.tools.image_collect_node:main',
        ],
    },
)