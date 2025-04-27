import argparse
import sys
import numpy as np
from vehicle import DriverType
from simulationGUI import SimulationGUI

def add_command_line_features(SimulationGUI):
    """
    Extends the SimulationGUI class with command line argument functionality.
    This is a monkey-patch approach that preserves the original class.
    """
    # Store the original __init__ method
    original_init = SimulationGUI.__init__

    def new_init(self):
        # Call the original __init__
        original_init(self)
        
        # Setup command line argument parsing
        self.setup_command_line_args()
        
        # Parse the arguments
        if len(sys.argv) > 1:  # Check if any arguments were provided
            self.parse_command_line_args()
            
            # If in command line mode, run the simulation automatically
            if self.command_line_mode:
                self.run_from_command_line()
    
    def setup_command_line_args(self):
        """Add command line argument parsing to the SimulationGUI class."""
        self.parser = argparse.ArgumentParser(description='Traffic Simulation Parameters')
        
        # Basic simulation parameters
        self.parser.add_argument('--road-length', type=float, help='Length of the road in meters')
        self.parser.add_argument('--lanes', type=int, help='Number of lanes')
        self.parser.add_argument('--distracted-percentage', type=float, help='Percentage of distracted drivers (0-100)')
        self.parser.add_argument('--sim-time', type=float, help='Simulation time in seconds')
        
        # Multiple simulation parameters
        self.parser.add_argument('--num-simulations', type=int, help='Number of simulations to run for each vehicle count')
        self.parser.add_argument('--vehicle-counts', type=str, help='Comma-separated list of vehicle counts (e.g., "10,20,30,40,50")')
        
        # Driver distribution
        self.parser.add_argument('--driver-distribution', type=str, 
                                help='Comma-separated driver type distribution (Aggressive,Normal,Cautious,Polite,Submissive)')
        
        # Simulation mode
        self.parser.add_argument('--mode', type=str, choices=['normal', 'no-animation', 'multiple'], 
                                help='Simulation mode: normal, no-animation, or multiple')
        
        # Optional vehicle addition
        self.parser.add_argument('--add-vehicle', action='store_true', help='Add a manual vehicle to the simulation')
        self.parser.add_argument('--vehicle-type', type=str, 
                                choices=['aggressive', 'normal', 'cautious', 'polite', 'submissive', 'obstacle'],
                                help='Driver type for the manual vehicle')
        self.parser.add_argument('--vehicle-lane', type=int, help='Lane for the manual vehicle (1-based)')
        self.parser.add_argument('--vehicle-position', type=float, help='Initial position for the manual vehicle')
        self.parser.add_argument('--vehicle-velocity', type=float, help='Desired velocity for the manual vehicle')
        self.parser.add_argument('--vehicle-deploy-time', type=float, help='Deployment time for the manual vehicle')
        self.parser.add_argument('--vehicle-distracted', action='store_true', help='Whether the manual vehicle is distracted')
        
        # Non-animated simulation steps
        self.parser.add_argument('--steps', type=int, help='Number of steps for non-animated simulation')
        
        # Flag to save animation
        self.parser.add_argument('--save-animation', action='store_true', help='Save the animation to a file')
        
        # Flag to indicate command line mode is active
        self.command_line_mode = False
    
    def parse_command_line_args(self):
        """Parse command line arguments and update simulation parameters."""
        args = self.parser.parse_args()
        
        # Check if we have enough arguments to run in command line mode
        required_args = ['mode']
        if all(hasattr(args, arg) and getattr(args, arg) is not None for arg in required_args):
            self.command_line_mode = True
        else:
            return  # Not enough arguments, continue with GUI mode
        
        # Update basic parameters if provided
        if args.road_length is not None:
            self.params['road_length'] = args.road_length
        
        if args.lanes is not None:
            self.params['lanes_count'] = args.lanes
        
        if args.distracted_percentage is not None:
            self.params['distracted_percentage'] = args.distracted_percentage
        
        if args.sim_time is not None:
            self.params['simulation_time'] = args.sim_time
        
        # Update multiple simulation parameters if provided
        if args.num_simulations is not None:
            self.num_simulations = args.num_simulations
        
        if args.vehicle_counts is not None:
            try:
                self.num_vehicles_array = [int(x.strip()) for x in args.vehicle_counts.split(',')]
            except ValueError:
                print("Error: Vehicle counts must be comma-separated integers")
                sys.exit(1)
        
        # Update driver distribution if provided
        if args.driver_distribution is not None:
            try:
                values = [float(x.strip()) for x in args.driver_distribution.split(',')]
                if len(values) != 5:
                    raise ValueError("Need exactly 5 values")
                
                if not 0.99 <= sum(values) <= 1.01:
                    raise ValueError("Values must sum to 1.0")
                
                if any(v < 0 for v in values):
                    raise ValueError("All values must be non-negative")
                
                self.params['driver_type_distribution'] = {
                    DriverType.AGGRESSIVE: values[0],
                    DriverType.NORMAL: values[1],
                    DriverType.CAUTIOUS: values[2],
                    DriverType.POLITE: values[3],
                    DriverType.SUBMISSIVE: values[4]
                }
            except Exception as e:
                print(f"Error parsing driver distribution: {e}")
                print("Using default distribution instead")
        
        # Update non-animated steps if provided
        if args.steps is not None:
            self.non_animated_steps = args.steps
        
        # Update save animation flag if provided
        if args.save_animation:
            self.save_animation = True
        
        # Add manual vehicle if requested
        if args.add_vehicle:
            driver_type_map = {
                'aggressive': DriverType.AGGRESSIVE,
                'normal': DriverType.NORMAL,
                'cautious': DriverType.CAUTIOUS,
                'polite': DriverType.POLITE,
                'submissive': DriverType.SUBMISSIVE,
                'obstacle': DriverType.OBSTACLE
            }
            
            # Check if required vehicle parameters are provided
            required_vehicle_args = ['vehicle_type', 'vehicle_lane', 'vehicle_position', 'vehicle_velocity']
            if all(hasattr(args, arg) and getattr(args, arg) is not None for arg in required_vehicle_args):
                vehicle_info = {
                    'driver_type': driver_type_map.get(args.vehicle_type, DriverType.NORMAL),
                    'lane': max(0, min(self.params['lanes_count'] - 1, args.vehicle_lane - 1)),  # Convert to 0-based and bound
                    'desired_velocity': args.vehicle_velocity if args.vehicle_type != 'obstacle' else 0,
                    'deployment_time': args.vehicle_deploy_time if args.vehicle_deploy_time is not None else 0,
                    'initial_position': args.vehicle_position,
                    'is_distracted': args.vehicle_distracted,
                }
                self.vehicle_deployments.append(vehicle_info)
            else:
                print("Warning: Incomplete vehicle parameters, manual vehicle not added")
    
    def run_from_command_line(self):
        """Run the simulation based on command line arguments."""
        args = self.parser.parse_args()
        
        # Create a simulation with the specified parameters
        simulation_mode = args.mode
        
        if simulation_mode == 'normal':
            # Create and run normal simulation
            self.simulation = self.create_simulation()
            self.simulation.run_simulation(save_animation=self.save_animation)
        
        elif simulation_mode == 'no-animation':
            # Run without animation
            self.simulation = self.create_simulation()
            self.simulation.debug = False
            self.simulation.run_without_animation(steps=self.non_animated_steps)
        
        elif simulation_mode == 'multiple':
            # Run multiple simulations
            self.run_multiple_simulations(None)  # None for the event parameter
        
        # Exit after running in command line mode
        sys.exit(0)
    
    # Replace the original __init__ with the new one
    SimulationGUI.__init__ = new_init
    
    # Add the new methods to the class
    SimulationGUI.setup_command_line_args = setup_command_line_args
    SimulationGUI.parse_command_line_args = parse_command_line_args
    SimulationGUI.run_from_command_line = run_from_command_line
    
    return SimulationGUI

# Extend the SimulationGUI class with command line features
ModifiedSimulationGUI = add_command_line_features(SimulationGUI)

# Create and run the GUI/simulation
if __name__ == "__main__":
    gui = ModifiedSimulationGUI()
    gui.setup_start_screen()  # This will only run if not in command line mode