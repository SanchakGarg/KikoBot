# KikoBot ROS 2 control

ROS 2 Cartesian pick-and-place controller for keyboard or joystick input, an exclusive input selector, a serial servo bridge, and a MuJoCo mirror.

## Package layout

This folder is a ROS 2 `ament_python` package. `model/` contains a copy of the existing robot model and mesh assets, so it can be built and installed independently.

## Build on the existing WSL ROS installation

Open the repository in WSL (for example, `/mnt/d/Ashoka University/Academics/Thesis/Kiko-Bot Controller`) and run:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd "<repository>/kikobot-ros"
rosdep install --from-paths . --ignore-src -r -y --skip-keys=ament_python
colcon build --symlink-install
source install/setup.bash
```

The controller package uses `rclpy`, `sensor_msgs`, `std_msgs`, and the standard `Joy` message. Install the optional MuJoCo dependencies into the Python environment used by ROS with `python3 -m pip install --user -r requirements.txt`. Serial motor mode requires the vendor `scservo_sdk` module used by the existing `Joint-Controller.py`. A graphical WSL session (such as WSLg) is needed for the viewer.

## Start simulation first

Terminal 1 (publishes simulated joint angles as `/joint_states`):

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash
ros2 run kikobot_ros mujoco
```

Terminal 2 (command mux and simulated motor plant):

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash
ros2 launch kikobot_ros control.launch.py input:=keyboard
```

Terminal 3 (keyboard reader; run it directly so it has an interactive TTY):

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash
ros2 run kikobot_ros keyboard_input --ros-args -p start_in_ready_pose:=true
```

For hardware, leave `start_in_ready_pose` false so the controller starts from measured angles.

Keyboard controls match `ManualPickPlace/pick_place_controller.py`: arrow keys move the Cartesian wrist-pivot target in XY, `Q` / `E` move it up/down, `R` / `F` rotate J5, `Z` / `X` rotate J6, `C` centres J5/J6, and `H` returns to the ready pose. Arrow and letter keys use terminal key-repeat behavior, so the node must run in an interactive TTY. In simulation, it starts at the ready pose; in hardware mode, it initializes from measured joint positions.

For an Xbox-compatible gamepad, install/start the ROS `joy` package (`ros2 run joy joy_node`) and launch `input:=joystick`. The left stick moves the Cartesian target, triggers move it down/up, the right stick rotates J5/J6, and A centres the wrist. Axis and button indices are configurable in `config/controls.yaml`. The arbiter selects exactly one input using `input:=keyboard` or `input:=joystick`.

The controller solves differential inverse kinematics for J1--J4 to position the `j5` wrist pivot, while J5/J6 remain manual. It uses the same 0.42 m XY radius, 0.06--0.55 m height bounds, joint limits, and J5 rotation-plane constraint as the existing MuJoCo pick-and-place script. The Cartesian target is converted to six joint angles in radians before reaching the mux and motor controller.

## Physical motors

Simulation mode never opens a serial port. For hardware, stop the MuJoCo process and launch with explicit hardware opt-in:

```bash
ros2 launch kikobot_ros control.launch.py input:=keyboard hardware:=true device:=/dev/ttyUSB0 baudrate:=1000000
```

The motor node defaults to torque disabled. After confirming the arm is mechanically clear, servo IDs match the six joints, the WSL serial device is correct, and the configured joint-to-servo mapping is calibrated, separately enable torque:

```bash
ros2 param set /motor_controller enable_torque true
```

`servo_zero_deg`, `servo_direction`, `max_speed_deg_s`, and `slow_zone_deg` are parameters. Servo raw position is mapped using the same 0..4095 ↔ -180..180 degree convention as the existing controller; set each zero and direction to match the physical assembly before enabling torque. The motor sends bounded intermediate setpoints at 20 Hz, with velocity tapering near each goal. No hardware values are assumed calibrated by this package. Torque state is preserved when the node exits to avoid unexpectedly releasing a loaded arm; explicitly disable it only when the arm can safely relax.

## Topics

| Topic | Type | Purpose |
|---|---|---|
| `/input/keyboard` | `Float64MultiArray` | Keyboard joint targets in radians |
| `/input/joystick` | `Float64MultiArray` | Joystick joint targets in radians |
| `/joint_commands` | `Float64MultiArray` | Selected joint targets in radians |
| `/joint_states` | `sensor_msgs/JointState` | Measured motor angles or MuJoCo state |

The motor and MuJoCo nodes both publish `/joint_states`; run one plant at a time. All command arrays are ordered J1 through J6. Targets are clipped to the limits in the existing MuJoCo model.
