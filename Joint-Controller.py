import tkinter as tk
from tkinter import ttk
import scservo_sdk as scs
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# CONFIG
# ============================================================

COM_PORT = "COM3"
BAUDRATE = 1000000

SERVO_IDS = [1, 2, 3, 4, 5, 6]


# ============================================================
# SERVO CONTROL TABLE
# ============================================================

ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42

ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_LOAD = 60

TORQUE_ON = 1
TORQUE_OFF = 0


# ============================================================
# POSITION / ANGLE
# ============================================================

RAW_MIN = 0
RAW_MAX = 4095

ANGLE_MIN = -180.0
ANGLE_MAX = 180.0

# Servo is corrected only when it moves farther
# than this from the target.
DEADBAND_DEG = 1.0


# ============================================================
# LOOP TIMING
# ============================================================

FEEDBACK_INTERVAL = 0.05       # 20 Hz
CORRECTION_INTERVAL = 0.20     # 200 ms
GRAPH_INTERVAL = 0.10          # 100 ms

GRAPH_SECONDS = 30


# ============================================================
# GLOBAL STATE
# ============================================================

targets = {
    servo_id: None
    for servo_id in SERVO_IDS
}

actual_angles = {
    servo_id: None
    for servo_id in SERVO_IDS
}

loads = {
    servo_id: 0
    for servo_id in SERVO_IDS
}

last_correction_time = {
    servo_id: 0.0
    for servo_id in SERVO_IDS
}

last_send_time = {
    servo_id: 0.0
    for servo_id in SERVO_IDS
}


# ============================================================
# TORQUE / HOLD STATE
# ============================================================

torque_enabled = False
hold_enabled = False

running = True


# ============================================================
# SERIAL CONNECTION
# ============================================================

port = scs.PortHandler(COM_PORT)
packet = scs.PacketHandler(0)


if not port.openPort():

    raise RuntimeError(
        f"Could not open serial port {COM_PORT}"
    )


if not port.setBaudRate(BAUDRATE):

    port.closePort()

    raise RuntimeError(
        f"Could not set baudrate to {BAUDRATE}"
    )


print("=" * 60)
print("Connected to servo bus")
print(f"Port     : {COM_PORT}")
print(f"Baudrate : {BAUDRATE}")
print("=" * 60)


# ============================================================
# ANGLE CONVERSION
# ============================================================

def raw_to_angle(raw):

    raw = max(
        RAW_MIN,
        min(RAW_MAX, raw)
    )

    angle = (
        (raw - RAW_MIN)
        * (ANGLE_MAX - ANGLE_MIN)
        / (RAW_MAX - RAW_MIN)
        + ANGLE_MIN
    )

    return angle


def angle_to_raw(angle):

    angle = max(
        ANGLE_MIN,
        min(ANGLE_MAX, angle)
    )

    raw = (
        (angle - ANGLE_MIN)
        * (RAW_MAX - RAW_MIN)
        / (ANGLE_MAX - ANGLE_MIN)
        + RAW_MIN
    )

    return int(round(raw))


# ============================================================
# READ PRESENT POSITION
# ============================================================

def read_position(servo_id):

    try:

        raw_position, result, error = packet.read2ByteTxRx(
            port,
            servo_id,
            ADDR_PRESENT_POSITION
        )

        if result != scs.COMM_SUCCESS:

            print(
                f"[ID {servo_id}] "
                f"Position read failed: "
                f"result={result}"
            )

            return None

        if error != 0:

            print(
                f"[ID {servo_id}] "
                f"Position read error: {error}"
            )

            return None

        return raw_position

    except Exception as e:

        print(
            f"[ID {servo_id}] "
            f"Position read exception: {e}"
        )

        return None


# ============================================================
# READ LOAD
# ============================================================

def read_load(servo_id):

    try:

        raw_load, result, error = packet.read2ByteTxRx(
            port,
            servo_id,
            ADDR_PRESENT_LOAD
        )

        if result != scs.COMM_SUCCESS:
            return None

        if error != 0:
            return None

        return raw_load

    except Exception:

        return None


# ============================================================
# INITIALIZE POSITIONS
# ============================================================

def initialize_positions():

    print()
    print("=" * 60)
    print("INITIAL SERVO POSITION READ")
    print("=" * 60)

    for servo_id in SERVO_IDS:

        raw_position = read_position(
            servo_id
        )

        if raw_position is None:

            print(
                f"ID {servo_id} | READ FAILED"
            )

            targets[servo_id] = None
            actual_angles[servo_id] = None

            continue

        angle = raw_to_angle(
            raw_position
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # The current physical position becomes the initial
        # target.
        #
        # This prevents the servo from moving toward 0°
        # when torque is enabled.
        # ----------------------------------------------------

        targets[servo_id] = angle
        actual_angles[servo_id] = angle

        print(
            f"ID {servo_id} | "
            f"Raw: {raw_position:4d} | "
            f"Angle: {angle:+8.2f}°"
        )

    print("=" * 60)
    print()


# ============================================================
# SEND POSITION
# ============================================================

def send_angle(servo_id, angle):

    if angle is None:
        return False

    try:

        raw_position = angle_to_raw(
            angle
        )

        result, error = packet.write2ByteTxRx(
            port,
            servo_id,
            ADDR_GOAL_POSITION,
            raw_position
        )

        if result != scs.COMM_SUCCESS:

            print(
                f"[ID {servo_id}] "
                f"Position write failed: "
                f"result={result}, error={error}"
            )

            return False

        if error != 0:

            print(
                f"[ID {servo_id}] "
                f"Position write error: {error}"
            )

            return False

        last_send_time[servo_id] = time.time()

        return True

    except Exception as e:

        print(
            f"[ID {servo_id}] "
            f"Position write exception: {e}"
        )

        return False


# ============================================================
# SET TORQUE
# ============================================================

def set_torque(servo_id, enabled):

    try:

        result, error = packet.write1ByteTxRx(
            port,
            servo_id,
            ADDR_TORQUE_ENABLE,
            TORQUE_ON if enabled else TORQUE_OFF
        )

        if result != scs.COMM_SUCCESS:

            print(
                f"[ID {servo_id}] "
                f"Torque command failed: "
                f"result={result}, error={error}"
            )

            return False

        if error != 0:

            print(
                f"[ID {servo_id}] "
                f"Torque error: {error}"
            )

            return False

        return True

    except Exception as e:

        print(
            f"[ID {servo_id}] "
            f"Torque exception: {e}"
        )

        return False


# ============================================================
# ENABLE ALL TORQUE
# ============================================================

def enable_all_torque():

    global torque_enabled
    global hold_enabled

    success = True

    print()
    print("Enabling torque...")

    for servo_id in SERVO_IDS:

        if not set_torque(
            servo_id,
            True
        ):

            success = False

    if not success:

        print(
            "Failed to enable torque on one or more servos."
        )

        return

    torque_enabled = True
    hold_enabled = True

    hold_button.config(
        text="Hold: ON"
    )

    torque_status_label.config(
        text="Torque: ON"
    )

    print(
        "Torque ENABLED"
    )

    print(
        "Closed-loop HOLD ENABLED"
    )


    # --------------------------------------------------------
    # Restore current targets
    # --------------------------------------------------------

    for servo_id in SERVO_IDS:

        target = targets[servo_id]

        if target is not None:

            send_angle(
                servo_id,
                target
            )


# ============================================================
# DISABLE ALL TORQUE
# ============================================================

def disable_all_torque():

    global torque_enabled
    global hold_enabled

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Stop the laptop-side closed loop BEFORE disabling
    # physical servo torque.
    # --------------------------------------------------------

    hold_enabled = False
    torque_enabled = False

    hold_button.config(
        text="Hold: OFF"
    )

    torque_status_label.config(
        text="Torque: OFF"
    )

    print()
    print(
        "Closed-loop HOLD DISABLED"
    )


    # --------------------------------------------------------
    # Disable physical torque
    # --------------------------------------------------------

    for servo_id in SERVO_IDS:

        set_torque(
            servo_id,
            False
        )

    print(
        "Torque DISABLED"
    )


# ============================================================
# HOLD TOGGLE
# ============================================================

def toggle_hold():

    global hold_enabled

    # --------------------------------------------------------
    # Can't enable hold if torque is off
    # --------------------------------------------------------

    if not torque_enabled:

        hold_enabled = False

        hold_button.config(
            text="Hold: OFF"
        )

        print(
            "Cannot enable HOLD while torque is OFF."
        )

        return


    hold_enabled = not hold_enabled


    if hold_enabled:

        hold_button.config(
            text="Hold: ON"
        )

        print(
            "Closed-loop HOLD enabled."
        )

        # Immediately command targets

        for servo_id in SERVO_IDS:

            target = targets[servo_id]

            if target is not None:

                send_angle(
                    servo_id,
                    target
                )

    else:

        hold_button.config(
            text="Hold: OFF"
        )

        print(
            "Closed-loop HOLD disabled."
        )


# ============================================================
# SLIDER CALLBACK
# ============================================================

def slider_changed(servo_id, value):

    try:

        angle = float(value)

    except ValueError:

        return


    # --------------------------------------------------------
    # Slider is TARGET.
    #
    # Actual position never modifies this.
    # --------------------------------------------------------

    targets[servo_id] = angle


    target_labels[
        servo_id
    ].config(
        text=f"Target: {angle:+.1f}°"
    )


    # --------------------------------------------------------
    # Only command the servo if torque is ON.
    # --------------------------------------------------------

    if torque_enabled:

        send_angle(
            servo_id,
            angle
        )


    status_labels[
        servo_id
    ].config(
        text="Target updated"
    )


# ============================================================
# GUI
# ============================================================

# ------------------------------------------------------------
# IMPORTANT:
#
# INITIAL POSITION READ MUST HAPPEN BEFORE CREATING SLIDERS.
# ------------------------------------------------------------

initialize_positions()


root = tk.Tk()

root.title(
    "Kiko-Bot Servo Controller"
)

root.geometry(
    "1700x950"
)

root.minsize(
    1300,
    750
)


# ============================================================
# MAIN LAYOUT
# ============================================================

main_frame = ttk.Frame(
    root
)

main_frame.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


# ============================================================
# LEFT SIDE
# ============================================================

control_frame = ttk.LabelFrame(
    main_frame,
    text="Servo Controls"
)

control_frame.pack(
    side="left",
    fill="both",
    expand=False,
    padx=(0, 10)
)


# ============================================================
# RIGHT SIDE
# ============================================================

graph_frame = ttk.LabelFrame(
    main_frame,
    text="Live Torque / Load"
)

graph_frame.pack(
    side="right",
    fill="both",
    expand=True
)


# ============================================================
# BUTTON BAR
# ============================================================

top_control = ttk.Frame(
    control_frame
)

top_control.pack(
    fill="x",
    padx=10,
    pady=10
)


enable_button = ttk.Button(
    top_control,
    text="Enable Torque",
    command=enable_all_torque
)

enable_button.pack(
    side="left",
    padx=5
)


disable_button = ttk.Button(
    top_control,
    text="Disable Torque",
    command=disable_all_torque
)

disable_button.pack(
    side="left",
    padx=5
)


hold_button = ttk.Button(
    top_control,
    text="Hold: OFF",
    command=toggle_hold
)

hold_button.pack(
    side="left",
    padx=5
)


torque_status_label = ttk.Label(
    top_control,
    text="Torque: OFF"
)

torque_status_label.pack(
    side="left",
    padx=15
)


# ============================================================
# SERVO CONTROLS
# ============================================================

target_labels = {}
actual_labels = {}
error_labels = {}
load_labels = {}
status_labels = {}
sliders = {}


for row, servo_id in enumerate(SERVO_IDS):

    frame = ttk.LabelFrame(
        control_frame,
        text=f"Servo {servo_id}"
    )

    frame.pack(
        fill="x",
        padx=10,
        pady=5
    )


    # --------------------------------------------------------
    # SERVO ID
    # --------------------------------------------------------

    ttk.Label(
        frame,
        text=f"ID {servo_id}",
        width=6
    ).grid(
        row=0,
        column=0,
        rowspan=2,
        padx=5,
        pady=5
    )


    # --------------------------------------------------------
    # SLIDER
    # --------------------------------------------------------

    initial_angle = targets[servo_id]

    if initial_angle is None:

        initial_angle = 0.0

        targets[servo_id] = 0.0


    slider = tk.Scale(
        frame,
        from_=ANGLE_MAX,
        to=ANGLE_MIN,
        orient="horizontal",
        resolution=0.1,
        length=400,
        command=lambda value, sid=servo_id:
            slider_changed(
                sid,
                value
            )
    )

    # --------------------------------------------------------
    # THIS IS NOW THE ACTUAL CURRENT POSITION
    # --------------------------------------------------------

    slider.set(
        initial_angle
    )

    slider.grid(
        row=0,
        column=1,
        columnspan=4,
        padx=5,
        pady=(5, 0)
    )

    sliders[servo_id] = slider


    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    target_label = ttk.Label(
        frame,
        text=f"Target: {initial_angle:+.1f}°",
        width=16
    )

    target_label.grid(
        row=1,
        column=1,
        padx=5,
        pady=5
    )

    target_labels[servo_id] = target_label


    # --------------------------------------------------------
    # ACTUAL
    # --------------------------------------------------------

    if actual_angles[servo_id] is None:

        actual_text = "Actual: ---"

    else:

        actual_text = (
            f"Actual: "
            f"{actual_angles[servo_id]:+.1f}°"
        )


    actual_label = ttk.Label(
        frame,
        text=actual_text,
        width=16
    )

    actual_label.grid(
        row=1,
        column=2,
        padx=5,
        pady=5
    )

    actual_labels[servo_id] = actual_label


    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    if actual_angles[servo_id] is None:

        error_text = "Error: ---"

    else:

        error_text = (
            f"Error: "
            f"{targets[servo_id] - actual_angles[servo_id]:+.1f}°"
        )


    error_label = ttk.Label(
        frame,
        text=error_text,
        width=14
    )

    error_label.grid(
        row=1,
        column=3,
        padx=5,
        pady=5
    )

    error_labels[servo_id] = error_label


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    load_label = ttk.Label(
        frame,
        text="Load: ---",
        width=12
    )

    load_label.grid(
        row=1,
        column=4,
        padx=5,
        pady=5
    )

    load_labels[servo_id] = load_label


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    status_label = ttk.Label(
        frame,
        text="Torque OFF",
        width=20
    )

    status_label.grid(
        row=2,
        column=1,
        columnspan=4,
        padx=5,
        pady=(0, 5)
    )

    status_labels[servo_id] = status_label


# ============================================================
# INFORMATION
# ============================================================

info_frame = ttk.LabelFrame(
    control_frame,
    text="Information"
)

info_frame.pack(
    fill="x",
    padx=10,
    pady=10
)


info_text = (
    "ANGLE MAPPING\n"
    "0 raw     = -180°\n"
    "2048 raw  = ~0°\n"
    "4095 raw  = +180°\n\n"
    "STARTUP\n"
    "Slider and target are initialized from\n"
    "the current physical servo position.\n\n"
    "CONTROL\n"
    "Slider = Target\n"
    "Actual = Live feedback\n"
    "Hold = Supervisory closed-loop correction\n\n"
    "LOAD\n"
    "Graph shows raw servo load,\n"
    "not calibrated N·m torque."
)


ttk.Label(
    info_frame,
    text=info_text,
    justify="left"
).pack(
    padx=10,
    pady=10
)


# ============================================================
# MATPLOTLIB GRAPHS
# ============================================================

fig, axes = plt.subplots(
    len(SERVO_IDS),
    1,
    figsize=(8, 10),
    sharex=True
)


if len(SERVO_IDS) == 1:
    axes = [axes]


graph_lines = {}


for ax, servo_id in zip(
    axes,
    SERVO_IDS
):

    ax.set_ylabel(
        f"ID {servo_id}"
    )

    ax.set_ylim(
        0,
        4095
    )

    ax.grid(
        True,
        alpha=0.3
    )

    line, = ax.plot(
        [],
        [],
        linewidth=1.5
    )

    graph_lines[
        servo_id
    ] = line


axes[-1].set_xlabel(
    "Time (seconds)"
)


fig.tight_layout()


canvas = FigureCanvasTkAgg(
    fig,
    master=graph_frame
)

canvas.get_tk_widget().pack(
    fill="both",
    expand=True
)


# ============================================================
# GRAPH DATA
# ============================================================

max_points = int(
    GRAPH_SECONDS /
    GRAPH_INTERVAL
)


graph_data = {
    servo_id: deque(
        maxlen=max_points
    )
    for servo_id in SERVO_IDS
}


graph_times = {
    servo_id: deque(
        maxlen=max_points
    )
    for servo_id in SERVO_IDS
}


graph_start_time = time.time()


# ============================================================
# FEEDBACK THREAD
# ============================================================

def feedback_loop():

    global running

    while running:

        loop_start = time.time()


        for servo_id in SERVO_IDS:

            if not running:
                break


            # ------------------------------------------------
            # READ ACTUAL POSITION
            # ------------------------------------------------

            raw_position = read_position(
                servo_id
            )

            if raw_position is not None:

                actual_angles[
                    servo_id
                ] = raw_to_angle(
                    raw_position
                )


            # ------------------------------------------------
            # READ LOAD
            # ------------------------------------------------

            raw_load = read_load(
                servo_id
            )

            if raw_load is not None:

                loads[
                    servo_id
                ] = raw_load


            # ------------------------------------------------
            # CLOSED LOOP
            # ------------------------------------------------

            if (
                torque_enabled
                and hold_enabled
                and raw_position is not None
            ):

                target = targets[
                    servo_id
                ]

                actual = actual_angles[
                    servo_id
                ]

                if (
                    target is not None
                    and actual is not None
                ):

                    error = (
                        target
                        - actual
                    )

                    now = time.time()


                    # ----------------------------------------
                    # Only correct outside deadband
                    # ----------------------------------------

                    if abs(error) > DEADBAND_DEG:

                        if (
                            now
                            - last_correction_time[
                                servo_id
                            ]
                            >= CORRECTION_INTERVAL
                        ):

                            send_angle(
                                servo_id,
                                target
                            )

                            last_correction_time[
                                servo_id
                            ] = now


        # ----------------------------------------------------
        # Maintain feedback frequency
        # ----------------------------------------------------

        elapsed = (
            time.time()
            - loop_start
        )

        sleep_time = (
            FEEDBACK_INTERVAL
            - elapsed
        )

        if sleep_time > 0:

            time.sleep(
                sleep_time
            )


# ============================================================
# GUI UPDATE
# ============================================================

def update_gui():

    if not running:
        return


    for servo_id in SERVO_IDS:

        actual = actual_angles[
            servo_id
        ]

        target = targets[
            servo_id
        ]

        load = loads[
            servo_id
        ]


        # ----------------------------------------------------
        # ACTUAL
        # ----------------------------------------------------

        if actual is None:

            actual_labels[
                servo_id
            ].config(
                text="Actual: ---"
            )

        else:

            actual_labels[
                servo_id
            ].config(
                text=f"Actual: {actual:+.1f}°"
            )


        # ----------------------------------------------------
        # ERROR
        # ----------------------------------------------------

        if (
            actual is None
            or target is None
        ):

            error_labels[
                servo_id
            ].config(
                text="Error: ---"
            )

        else:

            error = (
                target
                - actual
            )

            error_labels[
                servo_id
            ].config(
                text=f"Error: {error:+.1f}°"
            )


        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        load_labels[
            servo_id
        ].config(
            text=f"Load: {load}"
        )


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if not torque_enabled:

            status_labels[
                servo_id
            ].config(
                text="Torque OFF"
            )

        elif not hold_enabled:

            status_labels[
                servo_id
            ].config(
                text="Torque ON / Hold OFF"
            )

        elif actual is None:

            status_labels[
                servo_id
            ].config(
                text="Position read failed"
            )

        elif abs(
            target - actual
        ) <= DEADBAND_DEG:

            status_labels[
                servo_id
            ].config(
                text="Holding position"
            )

        else:

            status_labels[
                servo_id
            ].config(
                text="Correcting position"
            )


    root.after(
        100,
        update_gui
    )


# ============================================================
# GRAPH UPDATE
# ============================================================

def update_graph():

    if not running:
        return


    current_time = (
        time.time()
        - graph_start_time
    )


    for servo_id in SERVO_IDS:

        graph_times[
            servo_id
        ].append(
            current_time
        )

        graph_data[
            servo_id
        ].append(
            loads[
                servo_id
            ]
        )


        graph_lines[
            servo_id
        ].set_data(
            list(
                graph_times[
                    servo_id
                ]
            ),
            list(
                graph_data[
                    servo_id
                ]
            )
        )


    # --------------------------------------------------------
    # Moving time window
    # --------------------------------------------------------

    if current_time <= GRAPH_SECONDS:

        axes[-1].set_xlim(
            0,
            GRAPH_SECONDS
        )

    else:

        axes[-1].set_xlim(
            current_time - GRAPH_SECONDS,
            current_time
        )


    canvas.draw_idle()


    root.after(
        int(
            GRAPH_INTERVAL * 1000
        ),
        update_graph
    )


# ============================================================
# CLOSE
# ============================================================

def on_close():

    global running

    print()
    print("Closing controller...")

    running = False

    # --------------------------------------------------------
    # Do NOT disable torque automatically.
    #
    # This prevents the arm from suddenly falling when
    # the GUI is closed.
    # --------------------------------------------------------

    time.sleep(
        0.15
    )

    try:

        port.closePort()

    except Exception:

        pass


    root.destroy()

    print(
        "Controller closed."
    )


root.protocol(
    "WM_DELETE_WINDOW",
    on_close
)


# ============================================================
# START FEEDBACK THREAD
# ============================================================

feedback_thread = threading.Thread(
    target=feedback_loop,
    daemon=True
)

feedback_thread.start()


# ============================================================
# START GUI UPDATE LOOPS
# ============================================================

root.after(
    100,
    update_gui
)

root.after(
    int(
        GRAPH_INTERVAL * 1000
    ),
    update_graph
)


# ============================================================
# START GUI
# ============================================================

print(
    "GUI started."
)

print(
    "Torque is initially OFF."
)

print(
    "Sliders are initialized from current servo positions."
)

print(
    "Press Enable Torque to activate the servos."
)

print()

root.mainloop()


# ============================================================
# CLEANUP
# ============================================================

running = False

try:

    port.closePort()

except Exception:

    pass