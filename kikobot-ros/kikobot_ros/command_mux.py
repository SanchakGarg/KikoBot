import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from .common import clip_targets


class CommandMux(Node):
    def __init__(self):
        super().__init__("command_mux")
        self.declare_parameter("input", "keyboard")
        self.selected = str(self.get_parameter("input").value).lower()
        if self.selected not in ("keyboard", "joystick"):
            raise ValueError("input must be 'keyboard' or 'joystick'")
        self.publisher = self.create_publisher(Float64MultiArray, "/joint_commands", 10)
        for source in ("keyboard", "joystick"):
            self.create_subscription(
                Float64MultiArray, f"/input/{source}",
                lambda message, name=source: self.on_command(name, message), 10
            )
        self.get_logger().info(f"Exclusive command source: {self.selected}")

    def on_command(self, source, message):
        if source != self.selected:
            return
        if len(message.data) != 6:
            self.get_logger().warning("Discarding command that does not contain six joint targets")
            return
        output = Float64MultiArray()
        output.data = clip_targets(message.data)
        self.publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = CommandMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
