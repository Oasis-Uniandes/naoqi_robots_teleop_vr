import time
import argparse
import sys
import signal

from naoqi_teleop.shared_state import SharedState
from naoqi_teleop.robot_io import RobotIO
from naoqi_teleop.ik_solver import IKSolver
from naoqi_teleop.vr_server import VRServer
from naoqi_teleop.desktop_vis import DesktopVis

def main():
    parser = argparse.ArgumentParser(description="Standalone VR Teleoperator for NAO and Pepper")
    parser.add_argument("--robot-ip", type=str, default="127.0.0.1", help="Robot IP address")
    parser.add_argument("--robot-port", type=int, default=9559, help="Robot port (usually 9559)")
    parser.add_argument("--vuer-host", type=str, default="0.0.0.0", help="Vuer VR host")
    parser.add_argument("--vuer-port", type=int, default=8012, help="Vuer VR port")
    parser.add_argument("--viser-port", type=int, default=8080, help="Viser desktop port")
    args = parser.parse_args()

    state = SharedState()
    
    # Initialize all subsystems
    robot = RobotIO(state, ip=args.robot_ip, port=args.robot_port)
    ik = IKSolver(state)
    vr = VRServer(state, host=args.vuer_host, port=args.vuer_port)
    vis = DesktopVis(state, port=args.viser_port)
    
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
