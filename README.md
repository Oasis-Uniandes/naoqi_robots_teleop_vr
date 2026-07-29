# Standalone VR Teleoperator for NAO & Pepper

This project provides a standalone Python application for fully teleoperating Softbank robots (NAO and Pepper) using a WebXR VR headset. It operates independently of ROS or LeRobot, directly interfacing with the `qi` framework for high performance and low latency.

## Features

- **Universal Compatibility**: One codebase automatically detects whether it's connected to a NAO or Pepper robot.
- **VR Controller Teleoperation**: 
  - **Arms**: Full 6-DoF inverse kinematics (via `pyroki`) tracking your VR controllers.
  - **Grippers**: Variable trigger input maps directly to gripper open/close state.
  - **Base Movement**: Left joystick moves the robot (walk/roll) forward, back, and strafes left/right. Right joystick controls rotation.
  - **Body Tracking (Pepper Only)**: Rotating your body (headset yaw) naturally rotates Pepper's base to keep it aligned with you.
- **Head Tracking**: Headset pitch and yaw directly map to the robot's head joints.
- **Visuals**: Stacks both the top and bottom cameras from the robot and displays them natively in your VR environment.
- **Audio**: Hooks into the robot's microphones and streams them into the VR headset via an internal web stream.
- **Desktop Visualizer**: A Viser-powered desktop application provides a real-time 3D URDF rendering of the robot's current state alongside the camera feeds.
- **Safety First**: Automatically disables `ALAutonomousLife` and `ALBasicAwareness` to prevent the robot from fighting your inputs or tracking bystanders during teleoperation.

## Installation

1. Create a Python 3.10+ virtual environment.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have the `qi` library available in your environment (typically via the `pynaoqi` wheel).

## Configuration

The application uses `OmegaConf` for a highly flexible, hierarchical configuration system similar to LeRobot.

All defaults are defined in `configs/default.yaml`.

### Overriding Configurations
You can override any configuration directly from the command line when starting the application by specifying `path.to.key=value`:

```bash
# Example: Change the robot IP and decrease walking speed
python main.py robot.ip=192.168.1.100 teleop.joystick_walk_speed=0.1
```

### Configuration Keys Explained

#### `robot` group
- `robot.ip`: The IP address of the robot (default: `"127.0.0.1"`).
- `robot.port`: The NAOqi port (default: `9559`).
- `robot.loop_rate_hz`: How fast the main control loop runs to fetch cameras and send joints (default: `20`).
- `robot.startup_posture`: The posture the robot should transition to after disabling autonomy (e.g. `"StandInit"`). Set to empty string to skip.
- `robot.enable_audio`: Toggle microphone streaming to VR (default: `true`).
- `robot.enable_top_camera` / `enable_bot_camera`: Toggle individual camera streams (default: `true`).
- `robot.camera_resolution`: 0 = 160x120, 1 = 320x240, 2 = 640x480 (default: `2`).

#### `vr` group
- `vr.host`: Host address for the Vuer server (default: `"0.0.0.0"`).
- `vr.port`: Port for the VR WebXR interface (default: `8012`).
- `vr.viser_port`: Port for the Desktop 3D visualizer (default: `8080`).

#### `teleop` group
- `teleop.joystick_walk_speed`: Maximum translation speed in m/s (default: `0.2`).
- `teleop.joystick_rotate_speed`: Maximum rotational speed in rad/s (default: `0.3`).
- `teleop.body_yaw_gain`: Proportional gain for Pepper's base rotation when matching your headset's body rotation (default: `1.5`).
- `teleop.controller_pitch_offset_deg`: Pitch offset applied to the VR controllers to align them with the physical wrist joints of the robot. Quest controllers often point up relative to the wrist (default: `40.0`).
- `teleop.controller_z_offset_m`: Pushes the IK target backwards along the Z-axis to align the tracking origin with the physical wrist (default: `0.10`).
- `teleop.head_pitch_min` / `max`: Physical pitch limits for the head joints.
- `teleop.head_yaw_min` / `max`: Physical yaw limits for the head joints.

#### `ik` group
- `ik.loop_rate_hz`: How fast the inverse kinematics solver runs in its background thread (default: `100`).

## Running the Application

Simply execute the main script:
```bash
python main.py
```
The script will attempt to connect to the robot, disable its autonomy services, load the URDF, and host the VR and Viser servers. Open your WebXR-compatible browser to `https://<YOUR_IP>:8012` to enter VR mode!
