import time

import numpy as np
import scservo_sdk as scs

from ik_solver import IKSolver, BASE_ORIGIN
from pick_place import POINT_A, POINT_B, LIFT

COM_PORT = "COM3"
BAUDRATE = 1000000
SERVO_IDS = [1, 2, 3, 4, 5, 6]

ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_GOAL_SPEED = 46
ADDR_PRESENT_POSITION = 56
TORQUE_ON = 1
TORQUE_OFF = 0

# Raw speed units (servo-internal steps/sec). Lower = slower. 0 = unrestricted/max.
GOAL_SPEED = 200

RAW_MIN, RAW_MAX = 0, 4095
ANGLE_MIN, ANGLE_MAX = -180.0, 180.0

DEADBAND_DEG = 1.0
ARRIVE_TIMEOUT = 4.0
HOLD_DURATION = 0.5


def angle_to_raw(angle):
    angle = max(ANGLE_MIN, min(ANGLE_MAX, angle))
    raw = (angle - ANGLE_MIN) * (RAW_MAX - RAW_MIN) / (ANGLE_MAX - ANGLE_MIN) + RAW_MIN
    return int(round(raw))


def raw_to_angle(raw):
    raw = max(RAW_MIN, min(RAW_MAX, raw))
    return (raw - RAW_MIN) * (ANGLE_MAX - ANGLE_MIN) / (RAW_MAX - RAW_MIN) + ANGLE_MIN


def send_angles(port, packet, angles_deg):
    for servo_id, angle in zip(SERVO_IDS, angles_deg):
        packet.write2ByteTxRx(port, servo_id, ADDR_GOAL_POSITION, angle_to_raw(angle))


def read_angles(port, packet):
    angles = []
    for servo_id in SERVO_IDS:
        raw, result, error = packet.read2ByteTxRx(port, servo_id, ADDR_PRESENT_POSITION)
        angles.append(raw_to_angle(raw) if result == scs.COMM_SUCCESS and error == 0 else None)
    return angles


def wait_until_arrived(port, packet, target_deg):
    deadline = time.time() + ARRIVE_TIMEOUT
    while time.time() < deadline:
        current = read_angles(port, packet)
        if all(
            c is not None and abs(c - t) < DEADBAND_DEG
            for c, t in zip(current, target_deg)
        ):
            return
        time.sleep(0.05)


def solve_waypoint(ik, pos, label):
    point_dir = np.asarray(pos) - BASE_ORIGIN
    q, ok = ik.solve(pos, point_dir=point_dir)
    if not ok:
        print(f"{label} {pos} is not reachable")
    return q


def main():
    ik = IKSolver()

    waypoints = []
    for name, pos in (("A", POINT_A), ("B", POINT_B)):
        down = pos
        up = (pos[0], pos[1], pos[2] + LIFT)
        q_down = solve_waypoint(ik, down, f"{name} down")
        q_up = solve_waypoint(ik, up, f"{name} up")
        if q_down is None or q_up is None:
            continue
        waypoints.append(np.degrees(q_up))
        waypoints.append(np.degrees(q_down))
        waypoints.append(np.degrees(q_up))

    if len(waypoints) < 4:
        print("not enough reachable points, nothing to do")
        return

    port = scs.PortHandler(COM_PORT)
    packet = scs.PacketHandler(0)

    if not port.openPort() or not port.setBaudRate(BAUDRATE):
        print("could not open serial port")
        return

    for servo_id in SERVO_IDS:
        packet.write1ByteTxRx(port, servo_id, ADDR_TORQUE_ENABLE, TORQUE_ON)
        packet.write2ByteTxRx(port, servo_id, ADDR_GOAL_SPEED, GOAL_SPEED)

    try:
        while True:
            for target_deg in waypoints:
                send_angles(port, packet, target_deg)
                wait_until_arrived(port, packet, target_deg)
                time.sleep(HOLD_DURATION)
    except KeyboardInterrupt:
        pass
    finally:
        for servo_id in SERVO_IDS:
            packet.write1ByteTxRx(port, servo_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
        port.closePort()


if __name__ == "__main__":
    main()
