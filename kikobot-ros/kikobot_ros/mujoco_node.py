import os
import threading
import time

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .common import JOINT_NAMES, JOINT_LIMITS, READY_POSE_RAD, clip_targets


class MujocoMirror(Node):
    def __init__(self, model_path):
        super().__init__("mujoco_mirror")
        import mujoco
        import mujoco.viewer
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_NAMES]
        if any(index < 0 for index in self.joint_ids + self.actuator_ids):
            raise RuntimeError("MuJoCo model must contain J1..J6 joints and actuators")
        ready = clip_targets(READY_POSE_RAD)
        for joint_id, angle in zip(self.joint_ids, ready):
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = angle
        for actuator_id, angle in zip(self.actuator_ids, ready):
            self.data.ctrl[actuator_id] = angle
        mujoco.mj_forward(self.model, self.data)
        self.target = list(ready)
        self.state_sub = self.create_subscription(JointState, "/joint_states", self.on_state, 10)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.thread = threading.Thread(target=self.run_viewer, daemon=True)
        self.thread.start()
        self.get_logger().info(f"MuJoCo mirror loaded {model_path}; following /joint_states")

    def on_state(self, message):
        by_name = dict(zip(message.name, message.position))
        if all(name in by_name for name in JOINT_NAMES):
            self.target = clip_targets([by_name[name] for name in JOINT_NAMES])

    def run_viewer(self):
        mujoco = self.mujoco
        while rclpy.ok() and self.viewer.is_running():
            start = time.monotonic()
            for actuator, target in zip(self.actuator_ids, self.target):
                self.data.ctrl[actuator] = target
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(max(0.0, self.model.opt.timestep - (time.monotonic() - start)))

    def destroy_node(self):
        try:
            self.viewer.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    package_share = get_package_share_directory("kikobot_ros")
    model_path = os.path.join(package_share, "model", "scene.xml")
    node = MujocoMirror(model_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
