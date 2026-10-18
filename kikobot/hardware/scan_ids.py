import scservo_sdk as scs
import json
import time
from datetime import datetime

PORT = "COM3"
BAUDRATE = 1000000
OUTPUT_FILE = "servo_scan.json"

port = scs.PortHandler(PORT)
packet = scs.PacketHandler(0)

if not port.openPort():
    print("ERROR: Could not open port")
    exit()

if not port.setBaudRate(BAUDRATE):
    print("ERROR: Could not set baudrate")
    port.closePort()
    exit()

print("=" * 60)
print("SCServo Bus Scanner")
print("=" * 60)

servos = []

for servo_id in range(254):

    model, result, error = packet.ping(port, servo_id)

    if result != scs.COMM_SUCCESS:
        continue

    print(f"Found ID {servo_id} | Model {model}")

    servo = {
        "id": servo_id,
        "model_number": model
    }

    # Firmware version
    firmware, result_fw, error_fw = packet.read1ByteTxRx(
        port, servo_id, 6
    )

    if result_fw == scs.COMM_SUCCESS:
        servo["firmware_version"] = firmware

    # Position
    position, result_pos, error_pos = packet.read2ByteTxRx(
        port, servo_id, 56
    )

    if result_pos == scs.COMM_SUCCESS:
        servo["position_raw"] = position

    # Load
    load, result_load, error_load = packet.read2ByteTxRx(
        port, servo_id, 60
    )

    if result_load == scs.COMM_SUCCESS:
        servo["load_raw"] = load

    # Voltage
    voltage, result_voltage, error_voltage = packet.read1ByteTxRx(
        port, servo_id, 62
    )

    if result_voltage == scs.COMM_SUCCESS:
        servo["voltage"] = voltage / 10.0

    # Temperature
    temperature, result_temp, error_temp = packet.read1ByteTxRx(
        port, servo_id, 63
    )

    if result_temp == scs.COMM_SUCCESS:
        servo["temperature_c"] = temperature

    # Error/status
    status, result_status, error_status = packet.read1ByteTxRx(
        port, servo_id, 65
    )

    if result_status == scs.COMM_SUCCESS:
        servo["error_status"] = status

    servos.append(servo)

    time.sleep(0.02)

port.closePort()

# Create output
output = {
    "timestamp": datetime.now().isoformat(),
    "port": PORT,
    "baudrate": BAUDRATE,
    "servo_count": len(servos),
    "servos": servos
}

# Save JSON
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=4)

print()
print("=" * 60)
print(f"Found {len(servos)} servo(s)")
print(f"IDs: {[s['id'] for s in servos]}")
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 60)
