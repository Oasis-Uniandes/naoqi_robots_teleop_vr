import threading
import numpy as np
from typing import Dict, List, Optional, Tuple

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        
        # --- Robot Info & Sensors ---
        self.robot_type: str = "unknown"  # "nao" or "pepper"
        self.is_connected: bool = False
        
        # Sensors
        self.joint_angles: Dict[str, float] = {}
        self.camera_top: Optional[np.ndarray] = None
        self.camera_bottom: Optional[np.ndarray] = None
        self.audio_chunks: List[bytes] = []  # Ring buffer of recent audio
        
        # --- VR Inputs ---
        # 4x4 transformation matrices in Robot Base frame
        self.target_ik_left: Optional[np.ndarray] = None
        self.target_ik_right: Optional[np.ndarray] = None
        
        # Gripper close amount (0.0 = open, 1.0 = closed)
        self.target_grip_left: float = 0.0
        self.target_grip_right: float = 0.0
        
        # Walking velocity (vx, vy, vtheta)
        self.target_walk: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        
        # Head angles (pitch, yaw)
        self.target_head: Tuple[float, float] = (0.0, 0.0)
        
        # Body yaw delta (for Pepper to rotate its base to match the user's body)
        self.target_body_yaw_delta: float = 0.0
        
        # --- IK Outputs ---
        # Computed joint angles from the IK solver, to be sent to the robot
        self.ik_joints_left: Optional[Dict[str, float]] = None
        self.ik_joints_right: Optional[Dict[str, float]] = None

    def update_robot_sensors(self, joint_angles: Dict[str, float], cam_top: Optional[np.ndarray], cam_bot: Optional[np.ndarray]):
        with self.lock:
            self.joint_angles.update(joint_angles)
            if cam_top is not None:
                self.camera_top = cam_top
            if cam_bot is not None:
                self.camera_bottom = cam_bot

    def push_audio(self, chunk: bytes):
        with self.lock:
            self.audio_chunks.append(chunk)
            # Keep only the last 50 chunks to prevent memory bloat (e.g. ~1 second of audio)
            if len(self.audio_chunks) > 50:
                self.audio_chunks.pop(0)

    def pop_audio(self) -> bytes:
        """Returns all pending audio as a single byte string and clears the buffer."""
        with self.lock:
            data = b"".join(self.audio_chunks)
            self.audio_chunks.clear()
            return data
