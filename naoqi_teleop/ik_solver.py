import time
import threading
import numpy as np
import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description
import yourdfpy
from scipy.spatial.transform import Rotation as R

from .pyroki_snippets._solve_ik import solve_ik
from .shared_state import SharedState

class IKSolver:
    def __init__(self, shared_state: SharedState, cfg):
        self.shared_state = shared_state
        self.cfg = cfg
        self._running = False
        self._thread = None
        self.robot = None
        self.q_left = None
        self.q_right = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def _load_robot(self):
        print(f"IKSolver: Loading URDF for {self.shared_state.robot_type}...")
        try:
            if self.shared_state.robot_type == "pepper":
                urdf_path = self.cfg.robot.pepper_urdf_path
                if not urdf_path:
                    import robot_descriptions.pepper_description as desc
                    urdf_path = desc.URDF_PATH
            else:
                urdf_path = self.cfg.robot.nao_urdf_path
                if not urdf_path:
                    import robot_descriptions.nao_description as desc
                    urdf_path = desc.URDF_PATH
                    
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
                
            urdf = yourdfpy.URDF.load(str(urdf_path_obj), filename_handler=filename_handler)
            
            # Fix infinite limits for continuous joints before passing to pyroki
            for joint in urdf.joint_map.values():
                if joint.type in {"fixed", "floating", "planar"}:
                    continue
                if joint.limit is None:
                    joint.limit = yourdfpy.urdf.Limit(effort=0.0, velocity=np.pi, lower=None, upper=None)
                elif joint.limit.velocity is None:
                    joint.limit.velocity = np.pi

            self.robot = pk.Robot.from_urdf(urdf)
            
            self.q_left = np.array(self.robot.joint_var_cls(0).default_factory(), copy=True)
            self.q_right = np.array(self.robot.joint_var_cls(0).default_factory(), copy=True)
            
            # Optionally populate with current joint angles if available
            # We omit _sync_q_from_state as pyroki joint orders and q vectors differ
            print("IKSolver: Ready.")
        except Exception as e:
            print(f"IKSolver: Failed to load robot: {e}")

    def _loop(self):
        # Wait for robot type detection
        while self._running and self.shared_state.robot_type == "unknown":
            time.sleep(0.1)
            
        if not self._running:
            return
            
        self._load_robot()
        
        while self._running:
            if not self.robot:
                time.sleep(0.1)
                continue
                
            with self.shared_state.lock:
                target_l = self.shared_state.target_ik_left
                target_r = self.shared_state.target_ik_right
                
            if target_l is not None:
                self._solve_arm(target_l, "l_gripper", self.q_left, left=True)
                
            if target_r is not None:
                self._solve_arm(target_r, "r_gripper", self.q_right, left=False)
                
            time.sleep(1.0 / self.cfg.ik.loop_rate_hz)

    def _solve_arm(self, target_matrix: np.ndarray, link_name: str, q_state: np.ndarray, left: bool):
        prefix = "L" if left else "R"
        arm_joints = {
            f"{prefix}ShoulderPitch",
            f"{prefix}ShoulderRoll",
            f"{prefix}ElbowYaw",
            f"{prefix}ElbowRoll",
            f"{prefix}WristYaw"
        }
        joint_mask = np.array([
            1.0 if joint_name in arm_joints else 0.0
            for joint_name in self.robot.joints.actuated_names
        ])
        
        target_pos = target_matrix[:3, 3]
        target_wxyz = R.from_matrix(target_matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
        
        new_q = solve_ik(
            robot=self.robot,
            target_link_name=link_name,
            target_wxyz=target_wxyz,
            target_position=target_pos,
            joint_mask=joint_mask,
            prev_cfg=q_state
        )
        q_state[:] = new_q
        
        out_dict = {}
        for i, jname in enumerate(self.robot.joints.actuated_names):
            if jname in arm_joints:
                out_dict[jname] = float(q_state[i])
                
        with self.shared_state.lock:
            if left:
                self.shared_state.ik_joints_left = out_dict
            else:
                self.shared_state.ik_joints_right = out_dict
