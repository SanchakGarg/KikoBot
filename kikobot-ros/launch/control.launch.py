from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    input_source = LaunchConfiguration("input")
    keyboard_node = LaunchConfiguration("keyboard_node")
    hardware = LaunchConfiguration("hardware")
    device = LaunchConfiguration("device")
    baudrate = LaunchConfiguration("baudrate")
    start_ready = ParameterValue(PythonExpression(["'", hardware, "' == 'false'"]), value_type=bool)
    controls_file = os.path.join(get_package_share_directory("kikobot_ros"), "config", "controls.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("input", default_value="keyboard", choices=["keyboard", "joystick"]),
        DeclareLaunchArgument(
            "keyboard_node", default_value="false",
            description="Start the keyboard reader inside launch (normally run it directly in its own TTY)",
        ),
        DeclareLaunchArgument("hardware", default_value="false"),
        DeclareLaunchArgument("device", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("baudrate", default_value="1000000"),
        Node(package="kikobot_ros", executable="command_mux", name="command_mux",
             parameters=[{"input": ParameterValue(input_source, value_type=str)}], output="screen"),
        Node(package="kikobot_ros", executable="keyboard_input", name="pick_place_controller", output="screen",
             condition=IfCondition(PythonExpression([
                 "'", input_source, "' == 'keyboard' and '", keyboard_node, "' == 'true'"
             ])),
             parameters=[controls_file, {"start_in_ready_pose": start_ready}]),
        Node(package="kikobot_ros", executable="joystick_input", name="pick_place_controller", output="screen",
             condition=IfCondition(PythonExpression(["'", input_source, "' == 'joystick'"])),
             parameters=[controls_file, {"start_in_ready_pose": start_ready}]),
        Node(package="kikobot_ros", executable="motor_controller", name="motor_controller", output="screen",
             parameters=[controls_file, {
                 "hardware": ParameterValue(hardware, value_type=bool),
                 "device": ParameterValue(device, value_type=str),
                 "baudrate": ParameterValue(baudrate, value_type=int),
             }]),
    ])
