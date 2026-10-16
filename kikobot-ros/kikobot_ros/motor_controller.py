import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .common import JOINT_NAMES, JOINT_LIMITS, READY_POSE_RAD, clip_targets


class MotorController(Node):
    """SCServo bridge. Torque is opt-in; setpoints are slew limited and taper near target."""

    def __init__(self):
        super().__init__("motor_controller")
        self.declare_parameter("hardware", False)
        self.declare_parameter("device", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 1000000)
        self.declare_parameter("servo_ids", [1, 2, 3, 4, 5, 6])
        self.declare_parameter("servo_zero_deg", [0.0] * 6)
        self.declare_parameter("servo_direction", [1.0] * 6)
        self.declare_parameter("max_speed_deg_s", 30.0)
        self.declare_parameter("slow_zone_deg", 20.0)
        self.declare_parameter("min_speed_deg_s", 3.0)
        self.declare_parameter("enable_torque", False)
        self.hardware = bool(self.get_parameter("hardware").value)
        self.ids = list(self.get_parameter("servo_ids").value)
        self.zero = list(self.get_parameter("servo_zero_deg").value)
        self.direction = list(self.get_parameter("servo_direction").value)
        self.max_speed = float(self.get_parameter("max_speed_deg_s").value)
        self.slow_zone = float(self.get_parameter("slow_zone_deg").value)
        self.min_speed = float(self.get_parameter("min_speed_deg_s").value)
        self.enable_torque = bool(self.get_parameter("enable_torque").value)
        initial_pose = [0.0] * 6 if self.hardware else list(READY_POSE_RAD)
        self.target = list(initial_pose)
        self.commanded = list(initial_pose)
        self.measured = [0.0] * 6
        self.received_command = False
        self.have_servo_feedback = False
        self.feedback_seen = [False] * 6
        self.last_tick = time.monotonic()
        self.port = self.packet = self.scs = None
        self.sub = self.create_subscription(Float64MultiArray, "/joint_commands", self.on_command, 10)
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        # Hardware bus writes stay at 20 Hz; simulation needs a quicker command
        # handoff so a key press is not held behind the old 50 ms timer period.
        self.timer = self.create_timer(0.01 if not self.hardware else 0.05, self.tick)
        if self.hardware:
            self.open_bus()
        else:
            self.get_logger().info("Simulation-only mode: no serial device opened")

    def open_bus(self):
        try:
            import scservo_sdk as scs
            self.scs = scs
            self.port = scs.PortHandler(str(self.get_parameter("device").value))
            self.packet = scs.PacketHandler(0)
            if not self.port.openPort() or not self.port.setBaudRate(int(self.get_parameter("baudrate").value)):
                raise RuntimeError("Could not open serial bus or set baud rate")
            self.get_logger().warning("Serial connected; torque is DISABLED until enable_torque is explicitly set true")
            self.create_timer(0.5, self.sync_torque_parameter)
        except Exception as exc:
            self.hardware = False
            self.get_logger().error(f"Hardware initialization failed; staying disarmed: {exc}")

    def sync_torque_parameter(self):
        requested = bool(self.get_parameter("enable_torque").value)
        if requested == self.enable_torque or self.packet is None:
            return
        for servo_id in self.ids:
            result, error = self.packet.write1ByteTxRx(self.port, int(servo_id), 40, 1 if requested else 0)
            if result != self.scs.COMM_SUCCESS or error:
                self.get_logger().error(f"Torque state write failed for servo {servo_id}")
                requested = False
        self.enable_torque = requested
        self.get_logger().warning(f"Torque {'ENABLED' if requested else 'DISABLED'}")

    def on_command(self, message):
        if len(message.data) != 6:
            self.get_logger().warning("Discarding command that does not contain six joint targets")
            return
        self.target = clip_targets(message.data)
        self.received_command = True
        if not self.hardware:
            # The MuJoCo mirror follows /joint_states. Publish immediately rather
            # than waiting for the next simulation timer tick.
            self.commanded = list(self.target)
            self.publish_state()

    @staticmethod
    def raw_to_radians(raw, zero_deg, direction):
        servo_deg = (raw * 360.0 / 4095.0) - 180.0
        return math.radians((servo_deg - zero_deg) * direction)

    @staticmethod
    def radians_to_raw(angle, zero_deg, direction):
        servo_deg = math.degrees(angle) / direction + zero_deg
        servo_deg = max(-180.0, min(180.0, servo_deg))
        return int(round((servo_deg + 180.0) * 4095.0 / 360.0))

    def tick(self):
        now = time.monotonic()
        dt = min(0.1, max(0.0, now - self.last_tick))
        self.last_tick = now
        if self.hardware:
            for index, (target, (low, high)) in enumerate(zip(self.target, JOINT_LIMITS)):
                error = target - self.commanded[index]
                remaining_deg = abs(math.degrees(target - self.measured[index]))
                speed = self.max_speed
                if remaining_deg < self.slow_zone:
                    speed = max(self.min_speed, self.max_speed * remaining_deg / max(self.slow_zone, 1e-6))
                step = math.radians(speed) * dt
                self.commanded[index] += max(-step, min(step, error))
                self.commanded[index] = max(low, min(high, self.commanded[index]))
        else:
            # MuJoCo's position actuators already provide the motion dynamics.
            # A second software slew limiter made simulated motion lag and jitter.
            self.commanded = list(self.target)
        if self.hardware and self.packet is not None:
            self.read_and_write()
        self.publish_state()

    def read_and_write(self):
        for i, servo_id in enumerate(self.ids[:6]):
            raw, result, error = self.packet.read2ByteTxRx(self.port, int(servo_id), 56)
            if result == self.scs.COMM_SUCCESS and not error:
                self.measured[i] = self.raw_to_radians(raw, float(self.zero[i]), float(self.direction[i]))
                if not self.feedback_seen[i]:
                    self.commanded[i] = self.measured[i]
                    if not self.received_command:
                        self.target[i] = self.measured[i]
                    self.feedback_seen[i] = True
            if self.enable_torque:
                goal = self.radians_to_raw(self.commanded[i], float(self.zero[i]), float(self.direction[i]))
                self.packet.write2ByteTxRx(self.port, int(servo_id), 42, goal)
        self.have_servo_feedback = all(self.feedback_seen)
        self.measured = clip_targets(self.measured)

    def publish_state(self):
        if self.hardware and not self.have_servo_feedback:
            return
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = JOINT_NAMES
        message.position = self.measured if self.hardware else list(self.commanded)
        self.pub.publish(message)

    def destroy_node(self):
        if self.port is not None:
            try:
                # Preserve enabled torque on shutdown to avoid an unexpected arm drop.
                self.port.closePort()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
