from glob import glob
import os

from setuptools import find_packages, setup


package_name = "arm"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy", "ruckig", "codroid-robot-sdk==2.1.10"],
    zip_safe=True,
    description="ESTUN PBVS, target tracking, safety, and CRI control.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "estun_driver = arm.estun_driver:main",
            "estun_feedback_node = arm.estun_feedback_node:main",
        ]
    },
)
