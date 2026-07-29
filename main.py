import time
import argparse
import sys
import signal

from omegaconf import OmegaConf

from naoqi_teleop.shared_state import SharedState
from naoqi_teleop.robot_io import RobotIO
from naoqi_teleop.ik_solver import IKSolver
from naoqi_teleop.vr_server import VRServer
from naoqi_teleop.desktop_vis import DesktopVis

def main():
    # Load default configs
    base_cfg = OmegaConf.load("configs/default.yaml")
    
    # Allow overriding from CLI args (e.g. python main.py teleop.joystick_walk_speed=0.5)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_cfg, cli_cfg)
    
    print(f"Loaded Configuration:\n{OmegaConf.to_yaml(cfg)}")

    state = SharedState()
    
    # Initialize all subsystems with the config
    robot = RobotIO(state, cfg)
    ik = IKSolver(state, cfg)
    vr = VRServer(state, cfg)
    vis = DesktopVis(state, cfg)
    
    # Graceful shutdown handler
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        print("\nShutting down gracefully...")
        running = False
        
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Starting subsystems...")
    robot.connect()
    robot.start()
    ik.start()
    vis.start()
    vr.start()
    
    print("All subsystems running. Press Ctrl+C to exit.")
    
    try:
        while running:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
        
    print("Stopping subsystems...")
    robot.stop()
    ik.stop()
    vis.stop()
    vr.stop()
    print("Shutdown complete.")

if __name__ == "__main__":
    main()
