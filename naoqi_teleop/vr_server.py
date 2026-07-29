import asyncio
import io
import threading
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from starlette.responses import StreamingResponse

from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, MotionControllers, ImageBackground, Html

from .shared_state import SharedState

class VRServer:
    def __init__(self, shared_state: SharedState, host="0.0.0.0", port=8012):
        self.shared_state = shared_state
        self.host = host
        self.port = port
        
        self.app = Vuer(host=self.host, port=self.port)
        self._running = False
        self._thread = None
        
        self._last_body_yaw = 0.0

        self._setup_routes()
        self._setup_handlers()

    def _setup_routes(self):
        # Setup audio stream endpoint directly on the Vuer Starlette app
        @self.app.server.route("/audio_stream")
        async def audio_stream(request):
            async def generate_audio():
                # Yield a minimal WAV header (44 bytes) for 16kHz mono 16-bit PCM
                # We lie about the total size (use a huge number) for continuous streaming
                header = b'RIFF\xff\xff\xff\x7fWAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\xff\xff\xff\x7f'
                yield header
                while self._running:
                    chunk = self.shared_state.pop_audio()
                    if chunk:
                        yield chunk
                    else:
                        await asyncio.sleep(0.05)
            
            return StreamingResponse(generate_audio(), media_type="audio/wav")

    def _setup_handlers(self):
        @self.app.spawn(start=True)
        async def main(session: VuerSession):
            # Setup scene with controllers and an HTML audio element to play our stream
            session.set @ DefaultScene()
            session.upsert(MotionControllers(stream=True, key="motionControllers", left=True, right=True), to="bgChildren")
            session.upsert(Html(
                html="<audio autoplay src='/audio_stream'></audio>",
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
            T_offset = np.eye(4)
            T_offset[:3, :3] = R.from_euler("x", 40.0, degrees=True).as_matrix()
            T_offset[2, 3] = 0.10
            
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
                
                if len(left_axes) >= 4:
                    vx = -float(left_axes[3]) * 0.2  # up is -Y on stick -> forward
                    vy = -float(left_axes[2]) * 0.2  # left is -X on stick -> left
                else:
                    vx, vy = 0.0, 0.0
                    
                if len(right_axes) >= 4:
                    vtheta = -float(right_axes[2]) * 0.3 # left is -X on stick -> positive yaw
                else:
                    vtheta = 0.0
                    
                self.shared_state.target_walk = (vx, vy, vtheta)

        @self.app.add_handler("CAMERA_MOVE")
        async def on_camera_move(event, session: VuerSession):
            matrix = np.array(event.value["matrix"]).reshape(4, 4).T
            # Extract Pitch and Yaw
            euler = R.from_matrix(matrix[:3, :3]).as_euler("xyz")
            pitch = euler[0]
            yaw = euler[1]
            
            with self.shared_state.lock:
                # Update head pitch/yaw (limit to robot's physical bounds roughly)
                self.shared_state.target_head = (
                    np.clip(pitch, -0.6, 0.4),
                    np.clip(yaw, -2.0, 2.0)
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
                    distanceToCamera=1.5
                ), to="bgChildren")
                
            await asyncio.sleep(0.1) # 10Hz is plenty for VR camera stream

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

    def _run_server(self):
        import uvicorn
        # Run the vuer starlette app using uvicorn
        uvicorn.run(self.app.server, host=self.host, port=self.port, log_level="error")

    def stop(self):
        self._running = False
        # uvicorn doesn't shutdown gracefully from a thread easily without hacks, 
        # but since it's a daemon thread, main.py exiting will kill it.
