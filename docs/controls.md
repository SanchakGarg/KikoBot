# Kiko-Bot manual pick-and-place controller

`kikobot/sim/controller.py` is an interactive MuJoCo controller for the Kiko-Bot arm. It converts Cartesian commands into J1--J4 joint targets, using the `j5` body origin (the J5 wrist pivot) as its Cartesian end-effector point. J5 and J6 remain manual wrist joints. It is simulation-only: it does not send commands to the physical robot.

## Run it

From the repository root:

```powershell
python -m kikobot.sim.controller
```

The module loads `models/scene.xml` via `kikobot.SCENE_PATH`, so it runs from any working directory. It requires the same Python environment that runs the existing MuJoCo controller (`mujoco` and its `glfw` dependency).

## Controls

| Input | Action |
|---|---|
| Hold arrow keys | Continuously move the Cartesian target in world X/Y |
| Hold `Q` / `E` | Continuously move target up / down in world Z |
| Hold `R` / `F` | Continuously turn J5 positive / negative |
| Hold `Z` / `X` | Continuously turn J6 positive / negative |
| `C` | Centre J5 and J6 (0 rad) |
| `H` | Reset the robot and target to the controller's bent ready pose |

On a connected Xbox-compatible controller detected by GLFW:

| Input | Action |
|---|---|
| Left stick | Cartesian X/Y movement |
| Left trigger / right trigger | Z down / up |
| Right stick left/right | Manual J5 rotation |
| Right stick up/down | Manual J6 rotation |
| A | Centre J5 and J6 |

On Windows, held keys are read through the operating system's key-state API, so movement continues smoothly until release. The fallback on other systems uses normal keyboard auto-repeat. The script prints a one-line state report once per second and reports controller connection changes. Keyboard operation still works if the GLFW gamepad module is unavailable or no controller is connected.

The terminal status labels the J5 plane as `vertical` when manual J5 movement is allowed. `LOCKED` means the hinge axis is no longer horizontal, so the controller refuses manual J5 motion and any new automatic arm target that would worsen the plane constraint.

## Frame and target convention

The target is a point in the MuJoCo world frame, measured in metres. The base plate is fixed at the world origin and world Z points upward. The Cartesian end-effector point is the `j5` body origin, which is also the J5 pivot; the `eof` mesh and J6 are deliberately excluded from position IK. The controller starts in a bent ready pose (`J1..J6 = [0, -45, 105, 60, 0, 0]°`) that holds J5 and J6 at centre and makes the J5 hinge axis horizontal.

For safety and useful behaviour with this specific 540 mm-tall model, the target is limited to a 420 mm radius in XY and a Z range of 60--550 mm. Those limits prevent accidental requests below the base/floor and reduce unreachable targets. They are command limits, not a claim that every point inside them is reachable.

## Kinematics and IK algorithm

The model is a six-hinge serial chain:

```text
base → J1 → J2 → J3 → J4 → J5 → J6 → eof
```

The controller treats J1--J4 as the automatic arm and J5--J6 as manual wrist joints. Both wrist commands start at zero (the centre of their symmetric limits) and are never overwritten by the IK solver. `R`/`F` are therefore manual J5 controls, not an IK orientation command; `Z`/`X` are the matching manual J6 controls. A wrist-only command pauses Cartesian IK for that frame and adopts the resulting end-effector position as the new held point. Therefore a manual J5/J6 rotation cannot make J1--J4 compensate or swing the arm.

## IK logic, step by step

The physics state and the IK planning state have different jobs. MuJoCo's main `data` advances the simulated robot with gravity and actuator dynamics. A separate `ik_data` is a kinematic copy populated from the commanded joint angles. IK plans in that copy. This is important because small gravity-induced servo lag in the physics state must not be mistaken for a geometric J5-plane limit.

At each 50 Hz IK update, the controller:

1. Copies the current J1--J6 actuator targets into `ik_data`, then calls `mj_forward` to calculate the planned `j5` pivot position, the J5 world axis, and the geometric Jacobian.
2. Subtracts the planned `j5` position from the requested Cartesian point to get the XYZ error.
3. Gets translational Jacobian `Jp` and rotational Jacobian `Jr` with `mj_jacBody`, retaining only the J1--J4 columns. J5/J6 do not participate in the position solve.
4. Adds the plane equation. The J5 rotation plane is perpendicular to world XY when its normal (the J5 axis `a5`) is horizontal. The required condition is `a5,z = 0`, with a one-degree numerical tolerance.
5. Calculates the change in that axis component from joint motion using `d(a5,z)/dq = (a5 × world_z)^T Jr`. This creates a four-row task Jacobian: XYZ position plus one J5-plane equation, for the four automatic joints J1--J4.
6. Solves the four equations directly when the matrix is well-conditioned. Near a singularity it uses damped least squares:

   ```text
   dq = Jᵀ (J Jᵀ + λ² I)⁻¹ e,   λ = 0.035
   ```

7. Caps the largest joint update to 2 degrees and clips J1--J4 to the hard joint ranges from `robot.xml`.
8. Checks the nonlinear candidate pose in `ik_data`. If it exceeds the plane tolerance, halves the step and checks again. If no feasible step is found, it leaves the Cartesian target unchanged and holds the arm. When the user moves the target back toward the reachable region, IK can proceed from the retained target; it does not get stuck on a reset target.
9. Sends the accepted J1--J4 targets to the MuJoCo position actuators. J5/J6 targets are copied into the planning state but never changed by IK.

The main simulation then advances physics at the model timestep and displays the actual servo-driven pose. The target and actual pose can differ slightly while servos catch up under gravity. The plane status is based on the commanded kinematic pose, which is the pose the IK solver can directly enforce.

When only J5 or J6 is manually moved, Cartesian IK pauses for that frame and adopts the resulting pivot position as the held point. This prevents the automatic joints from trying to undo the wrist command.

## Important model detail

Do not use a plain `mj_resetData` for this model when expecting the neutral pose. The CAD export's joint `ref` values represent the lower-stop configuration. The script first resets using the model's explicit all-zero `home` keyframe, then applies its defined bent ready pose. A fully upright all-zero configuration is kinematically singular for some Cartesian directions, which is why it is not used as the interactive start pose.
