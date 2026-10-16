from glob import glob
from setuptools import setup

package_name = "kikobot_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/model", glob("model/*.xml")),
        ("share/" + package_name + "/model/assets", glob("model/assets/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="KikoBot Controller",
    maintainer_email="maintainer@example.com",
    description="ROS 2 control nodes for KikoBot",
    license="MIT",
    entry_points={"console_scripts": [
        "keyboard_input = kikobot_ros.keyboard_input:main",
        "joystick_input = kikobot_ros.joystick_input:main",
        "pick_place_controller = kikobot_ros.pick_place_controller:main",
        "command_mux = kikobot_ros.command_mux:main",
        "motor_controller = kikobot_ros.motor_controller:main",
        "mujoco = kikobot_ros.mujoco_node:main",
    ]},
)
