import time
import threading
import numpy as np
import viser
from viser.extras import ViserUrdf
import robot_descriptions

from .shared_state import SharedState

class DesktopVis:
    def __init__(self, shared_state: SharedState, cfg):
        self.shared_state = shared_state
        self.cfg = cfg
        self.port = cfg.vr.viser_port
        self.server = None
        self._running = False
        self._thread = None
        self.urdf = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def _load_urdf(self):
        if self.shared_state.robot_type == "pepper":
            import robot_descriptions.pepper_description as desc
            urdf_path = desc.URDF_PATH
        else:
            import robot_descriptions.nao_description as desc
            urdf_path = desc.URDF_PATH
            
        self.urdf = ViserUrdf(self.server, urdf_path)

    def _loop(self):
        self.server = viser.ViserServer(port=self.port)
        
        while self._running and self.shared_state.robot_type == "unknown":
            time.sleep(0.1)
            
        if not self._running:
            return
            
        self._load_urdf()
        
        while self._running:
            with self.shared_state.lock:
                angles = self.shared_state.joint_angles.copy()
                top = self.shared_state.camera_top
                bot = self.shared_state.camera_bottom
                
            # Update URDF
            if self.urdf and angles:
                self.urdf.update_cfg(angles)
                
            # Update cameras in Viser GUI (render as images floating or as GUI images)
            if top is not None:
                self.server.scene.add_image("cameras/top", top, render_width=1.0, position=(1, 0, 1.5), look_at=(0, 0, 0))
            if bot is not None:
                self.server.scene.add_image("cameras/bottom", bot, render_width=1.0, position=(1, 0, 0.5), look_at=(0, 0, 0))
                
            time.sleep(0.05)
