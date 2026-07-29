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
        print(f"DesktopVis: Loading URDF for {self.shared_state.robot_type}...")
        try:
            if self.shared_state.robot_type == "pepper":
                urdf_path = self.cfg.robot.pepper_urdf_path
                if not urdf_path:
                    import robot_descriptions.pepper_description as desc
                    urdf_path = desc.URDF_PATH
            else:
                urdf_path = self.cfg.robot.nao_urdf_path
                
            import yourdfpy
            from pathlib import Path
            urdf_path_obj = Path(urdf_path).expanduser().resolve()
            
            nao_description_root = urdf_path_obj.parents[2]
            package_roots = {
                "nao_description": nao_description_root,
                "nao_meshes": nao_description_root.parent.parent / "nao_meshes",
            }
            
            def filename_handler(fname):
                if fname.startswith("package://"):
                    package_name, _, rel_path = fname.removeprefix("package://").partition("/")
                    package_root = package_roots.get(package_name)
                    if package_root is not None:
                        return str((package_root / rel_path).resolve())
                return yourdfpy.filename_handler_magic(fname, dir=urdf_path_obj.parent)
                
            parsed_urdf = yourdfpy.URDF.load(str(urdf_path_obj), filename_handler=filename_handler)
            self.urdf = ViserUrdf(self.server, parsed_urdf)
        except Exception as e:
            import traceback
            print(f"DesktopVis: Failed to load URDF: {e}")
            print(traceback.format_exc())

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
                
            # Update cameras in Viser GUI
            if top is not None:
                self.server.scene.add_image("cameras/top", top, render_width=1.0, render_height=0.75, position=(1, 0, 1.5))
            if bot is not None:
                self.server.scene.add_image("cameras/bottom", bot, render_width=1.0, render_height=0.75, position=(1, 0, 0.5))
                
            time.sleep(0.05)
