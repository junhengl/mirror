#!/usr/bin/env python3
"""
Real-time Simulation Pipeline - Main Entry Point

Launches all nodes in proper order with synchronized data flow:
1. Simulation (physics + visualization)
2. Controller (1kHz PD control)
3. Retargeting (500Hz IK)
4. Body Tracking (30Hz ZED)

Run with sudo for ZED camera access:
    sudo /home/junhengl/body_tracking/.venv/bin/python -m real_time_sim.main

Or directly:
    sudo /home/junhengl/body_tracking/.venv/bin/python real_time_sim/main.py
"""

import os
import sys
import time
import signal
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_time_sim.config import PipelineConfig
from real_time_sim.shared_state import SharedState, RobotState
from real_time_sim.simulation import MuJoCoSimulation
from real_time_sim.nodes import BodyTrackingNode, RetargetingNode, ControllerNode


def print_banner():
    """Print startup banner."""
    print("=" * 70)
    print("  REAL-TIME HUMANOID BODY TRACKING SIMULATION")
    print("=" * 70)
    print("  Components:")
    print("    - MuJoCo Physics Simulation (1kHz)")
    print("    - PD Torque Control + FSM (1kHz)")
    print("    - IK-based Retargeting (500Hz)")
    print("    - ZED Body Tracking (30Hz)")
    print("=" * 70)
    print()


def print_timing_stats(shared: SharedState):
    """Print timing statistics."""
    stats = shared.get_timing_stats()
    print(f"\n[Timing] Sim: {stats['sim_hz']:.0f}Hz | "
          f"Control: {stats['control_hz']:.0f}Hz | "
          f"Retarget: {stats['retarget_hz']:.0f}Hz | "
          f"Tracking: {stats['tracking_hz']:.0f}Hz")


def main():
    parser = argparse.ArgumentParser(description="Real-time body tracking simulation")
    parser.add_argument('--no-camera', action='store_true',
                        help='Run without ZED camera (dummy tracking)')
    parser.add_argument('--no-render', action='store_true',
                        help='Run without visualization')
    parser.add_argument('--duration', type=float, default=None,
                        help='Simulation duration in seconds (default: infinite)')
    parser.add_argument('--hang-height', type=float, default=1.3,
                        help='Robot hanging height in meters (default: 1.3)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    args = parser.parse_args()
    
    print_banner()
    
    # Create configuration
    config = PipelineConfig()
    config.verbose = args.verbose
    config.sim.base_height = args.hang_height  # Set hanging height from CLI
    
    # Create shared state
    shared = SharedState()
    
    # Find model path (use torque-controlled model)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'westwood_robots', 'TH02-A7-torque.xml')
    if not os.path.exists(model_path):
        # Fallback to kinematic model if torque model doesn't exist
        model_path = os.path.join(base_dir, 'westwood_robots', 'TH02-A7-v2.xml')
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, 'themis', 'TH02-A7-v2.xml')
    
    if not os.path.exists(model_path):
        print(f"[Error] Model not found at: {model_path}")
        sys.exit(1)
    
    # Create components
    print("[Main] Initializing components...")
    
    simulation = MuJoCoSimulation(config, shared, model_path)
    simulation.render_enabled = not args.no_render
    
    # Set hanging height (modifies the welded base body position)
    simulation.set_base_height(args.hang_height)
    print(f"[Main] Robot hanging at height: {args.hang_height}m")
    
    controller = ControllerNode(config, shared)
    retargeter = RetargetingNode(config, shared)
    tracker = BodyTrackingNode(config, shared)
    
    # Signal handler for clean shutdown
    def signal_handler(sig, frame):
        print("\n[Main] Shutdown requested...")
        shared.request_shutdown()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start components in order (downstream to upstream)
        print("\n[Main] Starting nodes...")
        
        # 1. Start simulation visualization
        if simulation.render_enabled:
            simulation.start_viewer()
        
        # 2. Start controller (needs simulation feedback)
        controller.start()
        
        # 3. Start retargeting (needs controller running)
        retargeter.start()
        
        # 4. Start tracking (needs retargeting running)
        tracker.start()
        
        print("\n[Main] All nodes started. Running simulation...")
        print("[Main] Press Ctrl+C or close MuJoCo window to exit")
        print()
        
        # Timing stats
        last_stats_time = time.time()
        
        # Run simulation in main thread
        simulation.running = True
        simulation.wall_time_start = time.time()
        simulation.sim_time = 0.0
        simulation.step_count = 0
        
        last_render_time = 0.0
        render_interval = 1.0 / config.sim.render_fps
        
        while simulation.running:
            # Check shutdown
            if shared.is_shutdown_requested():
                break
            
            # Check viewer
            if simulation.render_enabled and not simulation.is_viewer_running():
                shared.request_shutdown()
                break
            
            # Check duration
            if args.duration is not None and simulation.sim_time >= args.duration:
                break
            
            # Step simulation once per loop iteration (real-time pacing)
            loop_start = time.perf_counter()
            simulation.step()
            
            # Update markers from retargeting output and IK solver state
            retarget = shared.get_retarget_output()
            
            # Get actual positions from IK solver's forward kinematics (not welded body frames)
            # This shows the dynamic IK solver positions, not static body attachments
            actual = {
                'hand_l': retarget.hand_l_act,
                'hand_r': retarget.hand_r_act,
                'elbow_l': retarget.elbow_l_act,
                'elbow_r': retarget.elbow_r_act,
            }
            
            # Build desired dict - use retargeting targets from ZED tracking
            if retarget.valid:
                desired = {
                    'hand_l': retarget.hand_l_des,
                    'hand_r': retarget.hand_r_des,
                    'elbow_l': retarget.elbow_l_des,
                    'elbow_r': retarget.elbow_r_des,
                }
            else:
                # No valid retargeting - show default pose end-effector positions as targets
                # These will be near actual when holding default pose
                desired = {}
            
            simulation.update_markers(desired, actual)
            
            # Render at specified framerate
            if time.time() - last_render_time >= render_interval:
                if simulation.render_enabled:
                    simulation.sync_viewer()
                last_render_time = time.time()
            
            # Print timing stats periodically
            if time.time() - last_stats_time >= 2.0:
                actual_hz = simulation.step_count / (time.time() - simulation.wall_time_start)
                shared.update_timing('sim', actual_hz)
                print_timing_stats(shared)
                
                # Print FSM state
                fsm_state = controller.get_fsm_state()
                tracking_error = controller.get_tracking_error()
                print(f"[State] FSM: {fsm_state.name} | "
                      f"Tracking Error: {tracking_error*180/3.14159:.1f}deg RMS | "
                      f"Retarget Valid: {retarget.valid}")
                
                last_stats_time = time.time()
            
            # Pace simulation to real-time (1kHz = 0.001s per step)
            elapsed = time.perf_counter() - loop_start
            sleep_time = config.sim.sim_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time * 0.9)  # Sleep slightly less, busy-wait the rest
                while time.perf_counter() - loop_start < config.sim.sim_dt:
                    pass  # Busy wait for precision
            
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")
    except Exception as e:
        print(f"\n[Main] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Shutdown all components
        print("\n[Main] Shutting down...")
        shared.request_shutdown()
        
        tracker.stop()
        retargeter.stop()
        controller.stop()
        simulation.stop()
        
        print("[Main] Shutdown complete")


if __name__ == "__main__":
    main()
