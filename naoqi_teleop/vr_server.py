import asyncio
import io
import time
import threading
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from http.server import HTTPServer, BaseHTTPRequestHandler

from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, MotionControllers, ImageBackground, Html, Head

from .shared_state import SharedState

class VRServer:
    def __init__(self, shared_state: SharedState, cfg):
        self.shared_state = shared_state
        self.cfg = cfg
        self.host = cfg.vr.host
        self.port = cfg.vr.port
        
        self.app = Vuer(
            host=self.host, 
            port=self.port,
            cert=cfg.vr.cert if cfg.vr.cert else None,
            key=cfg.vr.key if cfg.vr.key else None
        )
        self._running = False
        self._thread = None
        
        self._last_body_yaw = 0.0

        self._setup_routes()
        self._setup_handlers()

    def _setup_routes(self):
        # We start a separate threaded HTTP server for audio to avoid freezing Vuer's router
        class AudioHandler(BaseHTTPRequestHandler):
            def do_GET(req_self):
                if req_self.path == "/audio_stream":
                    req_self.send_response(200)
                    req_self.send_header('Content-Type', 'audio/wav')
                    req_self.send_header('Access-Control-Allow-Origin', '*')
                    req_self.end_headers()
                    
                    header = b'RIFF\xff\xff\xff\x7fWAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\xff\xff\xff\x7f'
                    try:
                        req_self.wfile.write(header)
                        while self._running:
                            chunk = self.shared_state.pop_audio()
                            if chunk:
                                req_self.wfile.write(chunk)
                                req_self.wfile.flush()
                            else:
                                time.sleep(0.05)
                    except Exception:
                        pass
                else:
                    req_self.send_response(404)
                    req_self.end_headers()
                    
            def log_message(self, format, *args):
                pass
                
        class ReusableTCPServer(HTTPServer):
            allow_reuse_address = True
                
        def run_audio_server():
            server = ReusableTCPServer(('0.0.0.0', self.port + 1), AudioHandler)
            while self._running:
                server.handle_request()
                
        self.audio_thread = threading.Thread(target=run_audio_server, daemon=True)

    def _setup_handlers(self):
        @self.app.spawn
        async def main(session: VuerSession):
            # Setup scene with controllers and an HTML audio element to play our stream
            session.set @ DefaultScene()
            session.upsert(MotionControllers(stream=True, key="motionControllers", left=True, right=True), to="bgChildren")
            session.upsert(Head(stream=True, fps=30), to="bgChildren")
            
            if self.cfg.robot.enable_audio:
                # Use the new separate audio port (Vuer port + 1)
                session.upsert(Html(
                    html=f"<audio autoplay src='http://localhost:{self.port + 1}/audio_stream'></audio>",
                    position=[0, 0, 0],
                    key="audio-player"
                ), to="bgChildren")
            
            # Start a background task to stream the camera feed to the VR background
            asyncio.create_task(self._camera_loop(session))

            while True:
                await asyncio.sleep(1.0)

        @self.app.add_handler("CONTROLLER_MOVE")
        async def on_controller_move(event, session: VuerSession):
            left_data = event.value.get("left")
            right_data = event.value.get("right")
            left_state = event.value.get("leftState", {})
            right_state = event.value.get("rightState", {})
            
            # 1. Update IK Targets (Apply controller offset to match wrist)
            pitch_offset = self.cfg.teleop.controller_pitch_offset_deg
            z_offset = self.cfg.teleop.controller_z_offset_m
            
            T_offset = np.eye(4)
            T_offset[:3, :3] = R.from_euler("x", pitch_offset, degrees=True).as_matrix()
            T_offset[2, 3] = z_offset
            
            with self.shared_state.lock:
                if left_data and len(left_data) >= 16:
                    mat = np.array(left_data[:16]).reshape(4, 4).T @ T_offset
                    self.shared_state.target_ik_left = mat
                if right_data and len(right_data) >= 16:
                    mat = np.array(right_data[:16]).reshape(4, 4).T @ T_offset
                    self.shared_state.target_ik_right = mat
                    
                # 2. Update Grippers
                self.shared_state.target_grip_left = float(left_state.get("triggerValue", 0.0))
                self.shared_state.target_grip_right = float(right_state.get("triggerValue", 0.0))
                
                # 3. Walking/Rolling (Left Joystick for X/Y, Right Joystick for Theta)
                # Gamepad axes: [touchpadX, touchpadY, thumbstickX, thumbstickY]
                # Usually thumbstick is indices 2 and 3.
                left_axes = left_state.get("axes", [0, 0, 0, 0])
                right_axes = right_state.get("axes", [0, 0, 0, 0])
                
                walk_speed = self.cfg.teleop.joystick_walk_speed
                rot_speed = self.cfg.teleop.joystick_rotate_speed
                
                if len(left_axes) >= 2:
                    vx = -float(left_axes[-1]) * walk_speed  # up is -Y on stick -> forward
                    vy = -float(left_axes[-2]) * walk_speed  # left is -X on stick -> left
                else:
                    vx, vy = 0.0, 0.0
                    
                if len(right_axes) >= 2:
                    vtheta = -float(right_axes[-2]) * rot_speed # left is -X on stick -> positive yaw
                else:
                    vtheta = 0.0
                    
                self.shared_state.target_walk = (vx, vy, vtheta)

        @self.app.add_handler("HEAD_MOVE")
        async def on_head_move(event, session: VuerSession):
            matrix = np.array(event.value["matrix"]).reshape(4, 4).T
            # Extract Pitch and Yaw
            euler = R.from_matrix(matrix[:3, :3]).as_euler("xyz")
            pitch = euler[0]
            yaw = euler[1]
            
            with self.shared_state.lock:
                # Update head pitch/yaw (limit to robot's physical bounds)
                self.shared_state.target_head = (
                    np.clip(pitch, self.cfg.teleop.head_pitch_min, self.cfg.teleop.head_pitch_max),
                    np.clip(yaw, self.cfg.teleop.head_yaw_min, self.cfg.teleop.head_yaw_max)
                )
                
                # For pepper, calculate delta body yaw
                if self.shared_state.robot_type == "pepper":
                    delta = yaw - self._last_body_yaw
                    # Avoid wrapping issues simply
                    if delta > np.pi: delta -= 2*np.pi
                    if delta < -np.pi: delta += 2*np.pi
                    self.shared_state.target_body_yaw_delta = delta
                    self._last_body_yaw = yaw

    async def _camera_loop(self, session: VuerSession):
        while self._running:
            with self.shared_state.lock:
                top = self.shared_state.camera_top
                bot = self.shared_state.camera_bottom
                
            if top is not None and bot is not None:
                # Stack top and bottom vertically
                combined = np.vstack([top, bot])
                # Downsample slightly for performance if needed, but 640x960 is fine
                img = Image.fromarray(combined)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=60)
                b64 = buf.getvalue()
                
                session.upsert(ImageBackground(
                    src=b64,
                    key="camera-feed",
                    distanceToCamera=4.0
                ), to="bgChildren")
                
            await asyncio.sleep(0.1) # 10Hz is plenty for VR camera stream

    def start(self):
        self._running = True
        self.audio_thread.start()
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

    def _run_server(self):
        # Run the vuer app using its built-in runner
        self.app.run()

    def stop(self):
        self._running = False
