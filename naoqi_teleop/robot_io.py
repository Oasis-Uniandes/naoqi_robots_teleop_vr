import time
import numpy as np
import threading
from typing import List

try:
    import qi
except ImportError:
    qi = None

from .shared_state import SharedState

class AudioProcessor:
    """A qi service to receive audio buffers from ALAudioDevice."""
    def __init__(self, shared_state: SharedState):
        self.shared_state = shared_state

    def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timeStamp, buffer):
        """Callback triggered by ALAudioDevice."""
        # buffer is a bytearray of PCM data (16-bit, 16000Hz usually)
        if buffer:
            self.shared_state.push_audio(bytes(buffer))

class RobotIO:
    def __init__(self, shared_state: SharedState, cfg):
        self.shared_state = shared_state
        self.cfg = cfg
        self.ip = cfg.robot.ip
        self.port = cfg.robot.port
        
        self.session = None
        self.memory = None
        self.motion = None
        self.video = None
        self.audio = None
        
        self.top_camera_client = None
        self.bot_camera_client = None
        self.audio_service_id = None
        
        self._running = False
        self._thread = None

    def connect(self):
        if not qi:
            print("qi module not found. Running in dummy mode.")
            return

        print(f"Connecting to robot at {self.ip}:{self.port}...")
        self.session = qi.Session()
        try:
            self.session.connect(f"tcp://{self.ip}:{self.port}")
        except RuntimeError as e:
            print(f"Failed to connect to robot: {e}")
            return
            
        self.shared_state.is_connected = True
        self.memory = self.session.service("ALMemory")
        self.motion = self.session.service("ALMotion")
        self.video = self.session.service("ALVideoDevice")
        self.audio = self.session.service("ALAudioDevice")
        
        # Detect robot type
        try:
            robot_type = self.memory.getData("RobotConfig/Body/Type")
            if "Pepper" in robot_type:
                self.shared_state.robot_type = "pepper"
            else:
                self.shared_state.robot_type = "nao"
        except Exception:
            self.shared_state.robot_type = "nao" # fallback
            
        print(f"Detected robot: {self.shared_state.robot_type}")
        
        # Shutdown Autonomy Services
        self._shutdown_autonomy()
        
        # Setup Video (ColorSpace 13 = BGR)
        res = self.cfg.robot.camera_resolution
        try:
            if self.cfg.robot.enable_top_camera:
                self.top_camera_client = self.video.subscribeCamera("top_cam_vr", 0, res, 13, 15)
            if self.cfg.robot.enable_bot_camera:
                self.bot_camera_client = self.video.subscribeCamera("bot_cam_vr", 1, res, 13, 15)
        except Exception as e:
            print(f"Failed to subscribe to cameras: {e}")
            
        # Setup Audio
        if self.cfg.robot.enable_audio:
            try:
                self.audio_processor = AudioProcessor(self.shared_state)
                self.audio_service_id = self.session.registerService("VRAudioProcessor", self.audio_processor)
                self.audio.setClientPreferences("VRAudioProcessor", 16000, 1, 0)
                self.audio.subscribe("VRAudioProcessor")
            except Exception as e:
                print(f"Failed to subscribe to audio: {e}")
            
        # Wake up
        if self.motion:
            self.motion.wakeUp()
            
    def _shutdown_autonomy(self):
        print("Checking autonomy services...")
        try:
            life = self.session.service("ALAutonomousLife")
            if life.getState() != "disabled":
                print("Disabling ALAutonomousLife... (robot may crouch)")
                life.setState("disabled")
                while life.getState() != "disabled":
                    time.sleep(0.5)
                print("ALAutonomousLife is now disabled.")
        except Exception as e:
            print(f"ALAutonomousLife not found or failed: {e}")
            
        try:
            awareness = self.session.service("ALBasicAwareness")
            if awareness.isEnabled():
                print("Disabling ALBasicAwareness...")
                awareness.setEnabled(False)
        except Exception:
            pass
            
        try:
            posture = self.session.service("ALRobotPosture")
            if self.cfg.robot.startup_posture:
                print(f"Going to startup posture: {self.cfg.robot.startup_posture}...")
                posture.goToPosture(self.cfg.robot.startup_posture, 0.5)
        except Exception as e:
            print(f"Failed to go to startup posture: {e}")
            
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
            
        if self.session:
            if self.top_camera_client:
                self.video.unsubscribe(self.top_camera_client)
            if self.bot_camera_client:
                self.video.unsubscribe(self.bot_camera_client)
            if self.audio_service_id:
                try:
                    self.audio.unsubscribe("VRAudioProcessor")
                    self.session.unregisterService(self.audio_service_id)
                except Exception:
                    pass
            self.session.close()

    def _control_loop(self):
        while self._running:
            if not self.shared_state.is_connected:
                time.sleep(0.1)
                continue
                
            self._fetch_sensors()
            self._apply_commands()
            
            # Loop rate driven by config
            time.sleep(1.0 / self.cfg.robot.loop_rate_hz)

    def _fetch_sensors(self):
        # 1. Fetch Cameras
        cam_top = None
        cam_bot = None
        try:
            if self.top_camera_client:
                img_data = self.video.getImageRemote(self.top_camera_client)
                if img_data:
                    # img_data[6] is the byte array
                    w, h = img_data[0], img_data[1]
                    cam_top = np.frombuffer(img_data[6], dtype=np.uint8).reshape((h, w, 3))
            
            if self.bot_camera_client:
                img_data = self.video.getImageRemote(self.bot_camera_client)
                if img_data:
                    w, h = img_data[0], img_data[1]
                    cam_bot = np.frombuffer(img_data[6], dtype=np.uint8).reshape((h, w, 3))
        except Exception as e:
            pass # Ignore occasional vision dropouts

        # 2. Fetch Joint Angles
        joint_names = ["Body"]
        try:
            angles = self.motion.getAngles(joint_names, True)
            # ALMotion.getAngles("Body") returns a list corresponding to getBodyNames()
            names = self.motion.getBodyNames("Body")
            joint_dict = dict(zip(names, angles))
            self.shared_state.update_robot_sensors(joint_dict, cam_top, cam_bot)
        except Exception:
            pass

    def _apply_commands(self):
        with self.shared_state.lock:
            ik_left = self.shared_state.ik_joints_left
            ik_right = self.shared_state.ik_joints_right
            grip_left = self.shared_state.target_grip_left
            grip_right = self.shared_state.target_grip_right
            walk = self.shared_state.target_walk
            head = self.shared_state.target_head
            body_yaw_delta = self.shared_state.target_body_yaw_delta
            
        if self.motion:
            # 1. Arms
            names = []
            angles = []
            if ik_left:
                names.extend(list(ik_left.keys()))
                angles.extend(list(ik_left.values()))
            if ik_right:
                names.extend(list(ik_right.keys()))
                angles.extend(list(ik_right.values()))
                
            # Grippers
            names.extend(["LHand", "RHand"])
            angles.extend([1.0 - grip_left, 1.0 - grip_right])
            
            # Head
            names.extend(["HeadPitch", "HeadYaw"])
            angles.extend([head[0], head[1]])
            
            if names:
                # Use a small fraction of max speed for smooth streaming
                self.motion.setAngles(names, angles, 0.15)
                
            # 2. Walking / Rolling
            vx, vy, vtheta = walk
            
            if self.shared_state.robot_type == "pepper":
                # For pepper, if there is a body yaw delta, add it to vtheta to align with the user
                # We can use a proportional gain to smoothly rotate the base.
                vtheta += body_yaw_delta * self.cfg.teleop.body_yaw_gain
                
            # Only send move commands if there is non-zero input or if we were just moving
            if abs(vx) > 0.05 or abs(vy) > 0.05 or abs(vtheta) > 0.05:
                self.motion.moveToward(vx, vy, vtheta)
            else:
                self.motion.moveToward(0.0, 0.0, 0.0)
