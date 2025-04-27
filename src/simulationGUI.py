import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Polygon, Circle
from matplotlib.widgets import Button, Slider, TextBox, RadioButtons, CheckButtons
import random
from collections import defaultdict
from enum import Enum
import pandas as pd
from vehicle import Vehicle, DriverType
from trafficSimulation import TrafficSimulation
import os
from tkinter import Tk, filedialog

class SimulationGUI:
    def __init__(self):
        self.params = {
            'output_dir': None,
            'road_length': 500,
            'lanes_count': 2,
            'n_vehicles': 30,
            'dt': 0.5,
            'simulation_time': 120,
            'animation_interval': 50,
            'distracted_percentage': 0,  # Adding default percentage of distracted drivers
            'driver_type_distribution': {  # Add default driver type distribution
                DriverType.AGGRESSIVE: 1,
                DriverType.NORMAL: 0,
                DriverType.CAUTIOUS: 0.0,
                DriverType.POLITE: 0.0,
                DriverType.SUBMISSIVE: 0.0
            }
        }
        self.simulation = None
        self.fig = None
        
        # Vehicle deployment list
        self.vehicle_deployments = []
        self.current_driver_type = DriverType.NORMAL
        self.current_lane = 0
        self.current_desired_velocity = 25  # m/s
        self.current_deployment_time = 0  # seconds
        self.current_initial_position = 0  # meters
        self.current_is_distracted = False  # Add distracted state
        
        # Non-animated simulation steps
        self.non_animated_steps = 10 * 100
        
        # Animation recording flag
        self.save_animation = False
        
        # Multiple simulations parameters
        self.num_simulations = 20  # Default number of simulations to run
        self.num_vehicles_array = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]  # Default array of vehicle counts
        self.num_vehicles_array.reverse()  # Reverse the order for better visualization

        
    def setup_start_screen(self):
        """Create and display the start screen with parameter controls and vehicle deployment list."""
        self.fig = plt.figure(figsize=(15, 10))
        
        # Define subplot grid with better spacing - changed to 2x3 layout
        gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1], width_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
        
        # Main parameter area
        ax_params = plt.subplot(gs[0, 0])
        ax_params.set_axis_off()
        ax_params.set_title('Simulation Parameters', fontsize=14)
        
        # Vehicle configuration area
        ax_vehicle_config = plt.subplot(gs[0, 1])
        ax_vehicle_config.set_axis_off()
        ax_vehicle_config.set_title('Vehicle Configuration', fontsize=14)
        
        # Vehicle list area - moved to right side
        ax_vehicle_list = plt.subplot(gs[0, 2])
        ax_vehicle_list.set_axis_off()
        ax_vehicle_list.set_title('Deployment Schedule', fontsize=14)
        
        # Button area
        ax_buttons = plt.subplot(gs[1, :])
        ax_buttons.set_axis_off()
        
        # Add main title
        self.fig.suptitle('Traffic Simulation Setup', fontsize=20, fontweight='bold')
        
        # ==== Simulation Parameters Section ====
        left_margin = 0.1
        param_width = 0.25
        param_height = 0.03
        param_spacing = 0.06
        param_top = 0.85
        
        # Create text boxes for parameters instead of sliders
        label_x = left_margin + 0.09
        textbox_width = 0.1
        textbox_left = left_margin + 0.07
        
        # Labels
        ax_params.text(label_x, param_top + 0.09, 'Road Length (m):', ha='right', va='center')
        ax_params.text(label_x, param_top + 0.09 - 2 * param_spacing, 'Lanes:', ha='right', va='center')
        ax_params.text(label_x, param_top + 0.09 - 4 * param_spacing, 'Initial Vehicles:', ha='right', va='center')
        ax_params.text(label_x, param_top + 0.09 - 6 * param_spacing, 'Sim Time (s):', ha='right', va='center')
        ax_params.text(label_x, param_top + 0.09 - 8 * param_spacing, 'Time Step (s):', ha='right', va='center')
        ax_params.text(label_x, param_top + 0.09 - 10 * param_spacing, 'Animation Speed:', ha='right', va='center')
        ax_params.text(label_x, param_top + 0.09 - 12 * param_spacing, '% Distracted:', ha='right', va='center')
        ax_params.text(label_x, param_top + 0.09 - 14 * param_spacing, 'Num Simulations:', ha='right', va='center')
        ax_params.text(label_x, param_top + 0.09 - 16 * param_spacing, 'Vehicle Counts:', ha='right', va='center')
        
        # Text boxes
        ax_length = plt.axes([textbox_left, param_top - 0.015, textbox_width, param_height])
        ax_lanes = plt.axes([textbox_left, param_top - param_spacing - 0.015, textbox_width, param_height])
        ax_vehicles = plt.axes([textbox_left, param_top - 2*param_spacing - 0.015, textbox_width, param_height])
        ax_simtime = plt.axes([textbox_left, param_top - 3*param_spacing - 0.015, textbox_width, param_height])
        ax_dt = plt.axes([textbox_left, param_top - 4*param_spacing - 0.015, textbox_width, param_height])
        ax_interval = plt.axes([textbox_left, param_top - 5*param_spacing - 0.015, textbox_width, param_height])
        ax_distracted_percentage = plt.axes([textbox_left, param_top - 6*param_spacing - 0.015, textbox_width, param_height])
        ax_num_simulations = plt.axes([textbox_left, param_top - 7*param_spacing - 0.015, textbox_width, param_height])
        ax_vehicle_counts = plt.axes([textbox_left, param_top - 8*param_spacing - 0.015, textbox_width, param_height])
        # Driver type distribution
        ax_params.text(label_x, param_top - 16 * param_spacing, 'Driver Dist [A,N,C,P,S]:', ha='right', va='center')
        ax_driver_dist = plt.axes([textbox_left, param_top - 9*param_spacing, textbox_width, param_height])

        # Create text boxes
        self.textbox_length = TextBox(ax_length, '', initial=str(self.params['road_length']))
        self.textbox_lanes = TextBox(ax_lanes, '', initial=str(self.params['lanes_count']))
        self.textbox_vehicles = TextBox(ax_vehicles, '', initial=str(self.params['n_vehicles']))
        self.textbox_simtime = TextBox(ax_simtime, '', initial=str(self.params['simulation_time']))
        self.textbox_dt = TextBox(ax_dt, '', initial=str(self.params['dt']))
        self.textbox_interval = TextBox(ax_interval, '', initial=str(self.params['animation_interval']))
        self.textbox_distracted_percentage = TextBox(
            ax_distracted_percentage, '', initial=str(self.params['distracted_percentage'])
        )
        self.textbox_num_simulations = TextBox(ax_num_simulations, '', initial=str(self.num_simulations))
        
        # For vehicle counts, join the array with commas
        initial_counts = ','.join(map(str, self.num_vehicles_array))
        self.textbox_vehicle_counts = TextBox(ax_vehicle_counts, '', initial=initial_counts)

        # Driver type distribution
        default_dist = self.create_distribution_string()
        self.textbox_driver_dist = TextBox(ax_driver_dist, '', initial=default_dist)
        self.textbox_driver_dist.on_submit(self.update_driver_distribution)
        
        # ==== Vehicle Configuration Section ====
        config_left = 0.4
        config_width = 0.25
        config_height = 0.03
        config_spacing = 0.06
        config_top = 0.85
        
        # Driver type selection - moved down and added obstacle option
        ax_driver_type = plt.axes([config_left, config_top - 0.2, config_width, 0.15])
        self.driver_type_selector = RadioButtons(
            ax_driver_type, 
            ['Aggressive', 'Normal', 'Cautious', 'Polite', 'Submissive', 'Obstacle'],
            active=1  # Default to Normal
        )
        self.driver_type_selector.on_clicked(self.update_driver_type)
        
        # Lane selection - start below the driver type selector
        vehicle_config_start = config_top - 0.25
        
        # Lane selection
        ax_lane = plt.axes([config_left, vehicle_config_start, config_width, config_height])
        self.textbox_lane = TextBox(ax_lane, 'Lane: ', initial='1')
        self.textbox_lane.on_submit(self.update_lane)
        
        # Initial position
        ax_position = plt.axes([config_left, vehicle_config_start - config_spacing, config_width, config_height])
        self.textbox_position = TextBox(ax_position, 'Initial Position (m): ', initial='0')
        self.textbox_position.on_submit(self.update_position)
        
        # Desired velocity
        ax_velocity = plt.axes([config_left, vehicle_config_start - 2*config_spacing, config_width, config_height])
        self.textbox_velocity = TextBox(ax_velocity, 'Desired Speed (m/s): ', initial=str(self.current_desired_velocity))
        self.textbox_velocity.on_submit(self.update_velocity)
        
        # Deployment time
        ax_deploy_time = plt.axes([config_left, vehicle_config_start - 3*config_spacing, config_width, config_height])
        self.textbox_deploy_time = TextBox(ax_deploy_time, 'Deploy Time (s): ', initial='0')
        self.textbox_deploy_time.on_submit(self.update_deploy_time)
        
        # Distracted checkbox - new addition
        ax_distracted = plt.axes([config_left, vehicle_config_start - 4*config_spacing, config_width, config_height])
        self.checkbox_distracted = CheckButtons(ax_distracted, ['Distracted Driver'], [False])
        self.checkbox_distracted.on_clicked(self.update_distracted)
        
        # Add vehicle button - moved down to not overlap with inputs
        ax_add_vehicle = plt.axes([config_left + 0.05, vehicle_config_start - 5*config_spacing - 0.05, 0.15, 0.05])
        self.button_add_vehicle = Button(ax_add_vehicle, 'Add Vehicle')
        self.button_add_vehicle.on_clicked(self.add_vehicle_to_list)
        
        # ==== Vehicle List Section (moved to right side) ====
        self.ax_vehicle_list = ax_vehicle_list
        self.update_vehicle_list_display()
        
        # Clear list button - positioned under the vehicle list on the right side
        ax_clear_list = plt.axes([0.75, 0.3, 0.15, 0.05])
        self.button_clear_list = Button(ax_clear_list, 'Clear List')
        self.button_clear_list.on_clicked(self.clear_vehicle_list)
        
        # ==== Non-animated Simulation Steps ====
        # Add a textbox for non-animated steps input
        ax_steps = plt.axes([0.35, 0.15, 0.1, 0.05])
        self.textbox_steps = TextBox(ax_steps, 'Steps: ', initial=str(self.non_animated_steps))
        self.textbox_steps.on_submit(self.update_steps)
        
        # ==== Save Animation Toggle ====
        # Add a checkbox for save animation toggle
        ax_save_anim = plt.axes([0.20, 0.15, 0.05, 0.07])
        self.checkbox_save_anim = CheckButtons(ax_save_anim, ['Save'], [False])
        self.checkbox_save_anim.on_clicked(self.update_save_animation)
        
        # ==== Control Buttons Section ====
        # Start button - moved left
        ax_start = plt.axes([0.25, 0.15, 0.15, 0.07])
        self.button_start = Button(ax_start, 'Start Simulation')
        self.button_start.on_clicked(self.start_simulation)
        
        # Run without animation button - placed to the right of Start button
        ax_no_anim = plt.axes([0.6, 0.15, 0.15, 0.07])
        self.button_no_animation = Button(ax_no_anim, 'Run Without Animation')
        self.button_no_animation.on_clicked(self.run_without_animation)
        
        # Run multiple simulations button
        ax_multi_sim = plt.axes([0.6, 0.05, 0.15, 0.07])
        self.button_multi_sim = Button(ax_multi_sim, 'Run Multiple Simulations')
        self.button_multi_sim.on_clicked(self.run_multiple_simulations)
        
        # Connect update functions
        self.textbox_length.on_submit(self.update_params)
        self.textbox_lanes.on_submit(self.update_params)
        self.textbox_vehicles.on_submit(self.update_params)
        self.textbox_simtime.on_submit(self.update_params)
        self.textbox_dt.on_submit(self.update_params)
        self.textbox_interval.on_submit(self.update_params)
        self.textbox_distracted_percentage.on_submit(self.update_params)
        self.textbox_num_simulations.on_submit(self.update_num_simulations)
        self.textbox_vehicle_counts.on_submit(self.update_vehicle_counts)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def create_distribution_string(self):
        """Create a string representation of the driver type distribution."""
        dist = self.params['driver_type_distribution']
        return f"{dist[DriverType.AGGRESSIVE]:.2f},{dist[DriverType.NORMAL]:.2f},{dist[DriverType.CAUTIOUS]:.2f},{dist[DriverType.POLITE]:.2f},{dist[DriverType.SUBMISSIVE]:.2f}"

    def update_driver_distribution(self, text):
        """Parse and update the driver type distribution from the text input."""
        try:
            # Parse comma-separated list of numbers
            values = [float(x.strip()) for x in text.split(',')]
            
            # Check if we have exactly 5 values (one for each driver type)
            if len(values) != 5:
                raise ValueError("Need exactly 5 values")
                
            # Check if values sum to approximately 1.0 (allow for small floating point errors)
            if not 0.99 <= sum(values) <= 1.01:
                raise ValueError("Values must sum to 1.0")
                
            # Check if all values are non-negative
            if any(v < 0 for v in values):
                raise ValueError("All values must be non-negative")
                
            # Update the driver type distribution
            self.params['driver_type_distribution'] = {
                DriverType.AGGRESSIVE: values[0],
                DriverType.NORMAL: values[1],
                DriverType.CAUTIOUS: values[2],
                DriverType.POLITE: values[3],
                DriverType.SUBMISSIVE: values[4]
            }
            
        except Exception as e:
            # Reset to default distribution
            self.params['driver_type_distribution'] = {
                DriverType.AGGRESSIVE: 0.1,
                DriverType.NORMAL: 0.6,
                DriverType.CAUTIOUS: 0.2,
                DriverType.POLITE: 0.05,
                DriverType.SUBMISSIVE: 0.05
            }
            self.textbox_driver_dist.set_val(self.create_distribution_string())
            print(f"Error parsing driver distribution: {e}")

    def update_num_simulations(self, text):
        """Update the number of simulations to run."""
        try:
            num = int(text)
            if num > 0:
                self.num_simulations = num
            else:
                self.textbox_num_simulations.set_val('5')
                self.num_simulations = 5
        except ValueError:
            self.textbox_num_simulations.set_val('5')
            self.num_simulations = 5
    
    def update_vehicle_counts(self, text):
        """Update the array of vehicle counts for multiple simulations."""
        try:
            # Parse comma-separated list of numbers
            counts = [int(x.strip()) for x in text.split(',')]
            if all(count > 0 for count in counts):
                self.num_vehicles_array = counts
            else:
                self.textbox_vehicle_counts.set_val('10,20,30,40,50')
                self.num_vehicles_array = [10, 20, 30, 40, 50]
        except ValueError:
            self.textbox_vehicle_counts.set_val('10,20,30,40,50')
            self.num_vehicles_array = [10, 20, 30, 40, 50]
    
    def update_distracted(self, label):
        """Update the distracted status when the checkbox is toggled."""
        self.current_is_distracted = self.checkbox_distracted.get_status()[0]
    
    def update_save_animation(self, label):
        """Update the save animation flag when the checkbox is toggled."""
        self.save_animation = self.checkbox_save_anim.get_status()[0]
    
    def update_steps(self, text):
        """Update the number of steps for non-animated simulation."""
        try:
            steps = int(text)
            if steps > 0:
                self.non_animated_steps = steps
            else:
                self.textbox_steps.set_val('10')
                self.non_animated_steps = 10
        except ValueError:
            self.textbox_steps.set_val('10')
            self.non_animated_steps = 10
    
    def update_driver_type(self, label):
        """Update the selected driver type."""
        driver_type_map = {
            'Aggressive': DriverType.AGGRESSIVE,
            'Normal': DriverType.NORMAL,
            'Cautious': DriverType.CAUTIOUS,
            'Polite': DriverType.POLITE,
            'Submissive': DriverType.SUBMISSIVE,
            'Obstacle': DriverType.OBSTACLE
        }
        self.current_driver_type = driver_type_map[label]
    
    def update_lane(self, text):
        """Update the selected lane."""
        try:
            lane = int(text)
            if 1 <= lane <= self.params['lanes_count']:
                self.current_lane = lane - 1  # Convert from 1-based UI to 0-based internal
            else:
                self.textbox_lane.set_val('1')
                self.current_lane = 0
        except ValueError:
            self.textbox_lane.set_val('1')
            self.current_lane = 0
    
    def update_position(self, text):
        """Update the initial position."""
        try:
            position = float(text)
            if 0 <= position <= self.params['road_length']:
                self.current_initial_position = position
            else:
                self.textbox_position.set_val('0')
                self.current_initial_position = 0
        except ValueError:
            self.textbox_position.set_val('0')
            self.current_initial_position = 0
    
    def update_velocity(self, text):
        """Update the desired velocity."""
        try:
            velocity = float(text)
            if 0 <= velocity <= 50:  # Reasonable limit
                self.current_desired_velocity = velocity
            else:
                self.textbox_velocity.set_val('25')
                self.current_desired_velocity = 25
        except ValueError:
            self.textbox_velocity.set_val('25')
            self.current_desired_velocity = 25
    
    def update_deploy_time(self, text):
        """Update the deployment time."""
        try:
            deploy_time = float(text)
            if 0 <= deploy_time <= self.params['simulation_time']:
                self.current_deployment_time = deploy_time
            else:
                self.textbox_deploy_time.set_val('0')
                self.current_deployment_time = 0
        except ValueError:
            self.textbox_deploy_time.set_val('0')
            self.current_deployment_time = 0
    
    def add_vehicle_to_list(self, event):
        """Add a vehicle to the deployment list."""
        vehicle_info = {
            'driver_type': self.current_driver_type,
            'lane': self.current_lane,
            'desired_velocity': self.current_desired_velocity if self.current_driver_type != DriverType.OBSTACLE else 0,
            'deployment_time': self.current_deployment_time,
            'initial_position': self.current_initial_position,
            'is_distracted': self.current_is_distracted,  # Added distracted state
        }
        
        self.vehicle_deployments.append(vehicle_info)
        self.update_vehicle_list_display()
    
    def clear_vehicle_list(self, event):
        """Clear the vehicle deployment list."""
        self.vehicle_deployments = []
        self.update_vehicle_list_display()
    
    def update_vehicle_list_display(self):
        """Update the display of the vehicle deployment list."""
        # Clear previous text
        for txt in self.ax_vehicle_list.texts:
            txt.remove()
        
        if not self.vehicle_deployments:
            self.ax_vehicle_list.text(0.5, 0.5, "No vehicles in deployment list", 
                                     ha='center', va='center', fontsize=12, 
                                     style='italic', color='gray')
            self.fig.canvas.draw_idle()
            return
        
        # Headers
        headers = ["#", "Type", "Lane", "Pos", "Speed", "Deploy", "Distracted"]
        header_pos = [0.05, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93]
        for i, header in enumerate(headers):
            self.ax_vehicle_list.text(header_pos[i], 0.95, header, 
                                     fontweight='bold', fontsize=10)
        
        # Driver type to display text mapping
        type_text = {
            DriverType.AGGRESSIVE: "Aggressive",
            DriverType.NORMAL: "Normal",
            DriverType.CAUTIOUS: "Cautious",
            DriverType.POLITE: "Polite",
            DriverType.SUBMISSIVE: "Submissive",
            DriverType.OBSTACLE: "Obstacle"
        }
        
        # List entries (display last 15 for space reasons)
        start_idx = max(0, len(self.vehicle_deployments) - 15)
        visible_deployments = self.vehicle_deployments[start_idx:]
        
        for i, vehicle in enumerate(visible_deployments):
            y_pos = 0.9 - (i+1) * 0.05
            
            # Row number (including offset if we're showing a partial list)
            self.ax_vehicle_list.text(header_pos[0], y_pos, f"{start_idx + i + 1}", fontsize=9)
            
            # Driver type
            self.ax_vehicle_list.text(header_pos[1], y_pos, type_text[vehicle['driver_type']], fontsize=9)
            
            # Lane (convert from 0-based to 1-based for display)
            self.ax_vehicle_list.text(header_pos[2], y_pos, f"{vehicle['lane'] + 1}", fontsize=9)
            
            # Initial position
            self.ax_vehicle_list.text(header_pos[3], y_pos, f"{vehicle['initial_position']}", fontsize=9)
            
            # Speed
            self.ax_vehicle_list.text(header_pos[4], y_pos, f"{vehicle['desired_velocity']}", fontsize=9)
            
            # Deployment time
            self.ax_vehicle_list.text(header_pos[5], y_pos, f"{vehicle['deployment_time']}", fontsize=9)
            
            # Distracted status
            distracted_text = "Yes" if vehicle.get('is_distracted', False) else "No"
            self.ax_vehicle_list.text(header_pos[6], y_pos, distracted_text, fontsize=9)
        
        # If we're showing a partial list, indicate how many more entries exist
        if start_idx > 0:
            self.ax_vehicle_list.text(0.5, 0.15, f"(+ {start_idx} more vehicles not shown)", 
                                     ha='center', fontsize=9, style='italic')
        
        self.fig.canvas.draw_idle()
        
    def update_params(self, text):
        """Update parameters when text boxes change."""
        try:
            self.params['road_length'] = float(self.textbox_length.text)
            self.params['lanes_count'] = int(float(self.textbox_lanes.text))
            self.params['n_vehicles'] = int(float(self.textbox_vehicles.text))
            self.params['simulation_time'] = float(self.textbox_simtime.text)
            self.params['dt'] = float(self.textbox_dt.text)
            self.params['animation_interval'] = float(self.textbox_interval.text)
            
            # Parse and validate distracted percentage
            distracted_pct = float(self.textbox_distracted_percentage.text)
            if 0 <= distracted_pct <= 100:
                self.params['distracted_percentage'] = distracted_pct
            else:
                self.textbox_distracted_percentage.set_val('0')
                self.params['distracted_percentage'] = 0
            
            # Note: Driver distribution is handled separately in update_driver_distribution
            
            # Validate and correct values if needed
            if self.params['lanes_count'] < 1:
                self.params['lanes_count'] = 1
                self.textbox_lanes.set_val('1')
                
            # Validate current lane selection against new lane count
            current_lane_value = int(self.textbox_lane.text) if self.textbox_lane.text.isdigit() else 1
            if current_lane_value > self.params['lanes_count']:
                self.textbox_lane.set_val(str(self.params['lanes_count']))
                self.current_lane = self.params['lanes_count'] - 1
                
            # Validate current deploy time against new simulation time
            current_deploy_time = float(self.textbox_deploy_time.text) if self.textbox_deploy_time.text.replace('.', '', 1).isdigit() else 0
            if current_deploy_time > self.params['simulation_time']:
                self.textbox_deploy_time.set_val(str(self.params['simulation_time']))
                self.current_deployment_time = self.params['simulation_time']
                
            # Validate current position against new road length
            current_position = float(self.textbox_position.text) if self.textbox_position.text.replace('.', '', 1).isdigit() else 0
            if current_position > self.params['road_length']:
                self.textbox_position.set_val(str(self.params['road_length']))
                self.current_initial_position = self.params['road_length']
                    
        except ValueError:
            # Reset to defaults if invalid input
            self.textbox_length.set_val(str(1000))
            self.textbox_lanes.set_val(str(3))
            self.textbox_vehicles.set_val(str(30))
            self.textbox_simtime.set_val(str(120))
            self.textbox_dt.set_val(str(0.5))
            self.textbox_interval.set_val(str(50))
            self.textbox_distracted_percentage.set_val(str(0))
            
            # Also reset the driver distribution
            default_dist = "0.10,0.60,0.20,0.05,0.05"
            self.textbox_driver_dist.set_val(default_dist)
            
            self.params = {
                'road_length': 1000,
                'lanes_count': 3,
                'n_vehicles': 30,
                'dt': 0.5,
                'simulation_time': 120,
                'animation_interval': 50,
                'distracted_percentage': 0,
                'driver_type_distribution': {
                    DriverType.AGGRESSIVE: 0.1,
                    DriverType.NORMAL: 0.6,
                    DriverType.CAUTIOUS: 0.2,
                    DriverType.POLITE: 0.05,
                    DriverType.SUBMISSIVE: 0.05
                }
            }
        
    def create_simulation(self, num_vehicles=None, to_print=True):
        """Create a simulation instance with the current parameters."""
        # Create simulation with selected parameters
        if num_vehicles is None:
            num_vehicles = self.params['n_vehicles']
            
        simulation = TrafficSimulation(
            road_length=self.params['road_length'],
            lanes_count=self.params['lanes_count'],
            n_vehicles=num_vehicles,
            dt=self.params['dt'],
            simulation_time=self.params['simulation_time'],
            animation_interval=self.params['animation_interval'],
            distracted_percentage=self.params['distracted_percentage'],
            driver_distribution=self.params['driver_type_distribution'],
            to_print=to_print
        )
        
        # Add the vehicle deployment schedule to the simulation
        simulation.scheduled_vehicles = self.vehicle_deployments.copy()
        # Store original scheduled vehicles for reset
        simulation.original_scheduled_vehicles = self.vehicle_deployments.copy()
        
        return simulation
    
    def start_simulation(self, event):
        """Start the simulation with the selected parameters and vehicle deployments."""
        plt.close(self.fig)  # Close start screen
        
        # Create and run simulation with selected parameters
        self.simulation = self.create_simulation()
        self.simulation.run_simulation(save_animation=self.save_animation)
    
    def run_without_animation(self, event):
        """Run the simulation without animation for a specified number of steps."""
        plt.close(self.fig)  # Close start screen
        
        # Create simulation with selected parameters
        self.simulation = self.create_simulation()
        
        # Enable debug mode to see detailed information 
        self.simulation.debug = False
        
        # Run the simulation without animation for the specified number of steps
        self.simulation.run_without_animation(steps=self.non_animated_steps)
        
    def run_multiple_simulations(self, event):
        """
        Run multiple simulations with different vehicle counts and collect statistics.
        Each simulation will run for 1000 steps and the average speed will be recorded.
        Results will be saved to two sheets in one Excel file.
        All outputs are stored in a dedicated folder for this simulation run.
        """
        plt.close(self.fig)  # Close start screen
        
        # Create a dedicated folder for this simulation run
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        folder_name = f'results/2L/NA/simulation_run_{timestamp}'
        if self.params['output_dir']:
            folder_name = self.params['output_dir']
        os.makedirs(folder_name, exist_ok=True)
        os.makedirs(folder_name, exist_ok=True)
        print(f"Creating output folder: {folder_name}")
        
        print(f"Starting multiple simulations: {self.num_simulations} runs for each of {len(self.num_vehicles_array)} vehicle counts")
        print(f"Vehicle counts: {self.num_vehicles_array}")
        
        # Set up results storage
        all_results = []
        steps_per_simulation = 500  # Fixed at 1000 steps per simulation
        
        # Run simulations for each vehicle count
        for vehicle_count in self.num_vehicles_array:
            print(f"\nRunning simulations with {vehicle_count} vehicles...")
            
            # Calculate density: vehicles / (road length * lanes)
            density = vehicle_count / (self.params['road_length'] * self.params['lanes_count'])
            
            # Run multiple simulations with the same parameters
            for sim_num in range(self.num_simulations):
                print(f"  Simulation {sim_num + 1}/{self.num_simulations}...")
                
                # Create new simulation with current vehicle count
                simulation = self.create_simulation(num_vehicles=vehicle_count, to_print=False)
                
                # Run without animation
                avg_speed = simulation.run_without_animation(steps=steps_per_simulation)
                
                # Calculate flow: density * average speed
                flow = density * avg_speed
                
                # Store result
                result = {
                    'Simulation Number': sim_num + 1,
                    'Number of Vehicles': vehicle_count,
                    'Number of Lanes': self.params['lanes_count'],
                    'Road Length': self.params['road_length'],
                    'Simulation Time (s)': self.params['simulation_time'],
                    'Time Step (s)': self.params['dt'],
                    'Animation Interval (ms)': self.params['animation_interval'],
                    'Percentage of Distracted Vehicles': self.params['distracted_percentage'],
                    'Aggressive %': self.params['driver_type_distribution'][DriverType.AGGRESSIVE] * 100,
                    'Normal %': self.params['driver_type_distribution'][DriverType.NORMAL] * 100,
                    'Cautious %': self.params['driver_type_distribution'][DriverType.CAUTIOUS] * 100,
                    'Polite %': self.params['driver_type_distribution'][DriverType.POLITE] * 100,
                    'Submissive %': self.params['driver_type_distribution'][DriverType.SUBMISSIVE] * 100,
                    'Average Speed': avg_speed,
                    'Density': density,
                    'Flow': flow
                }
                all_results.append(result)
                
                print(f"    Average speed: {avg_speed:.2f} m/s")
                print(f"    Density: {density:.4f} vehicles/m")
                print(f"    Flow: {flow:.4f} vehicles/s")
        
        # Create detailed results DataFrame
        df_detailed = pd.DataFrame(all_results)
        
        # Create summary results DataFrame by grouping by vehicle count
        df_summary = df_detailed.groupby(['Number of Vehicles', 'Number of Lanes', 'Road Length', 
                                        'Percentage of Distracted Vehicles']).agg(
            Average_Speed=('Average Speed', 'mean'),
            Variance=('Average Speed', 'var'),
            Std_Dev=('Average Speed', 'std'),
            Min_Speed=('Average Speed', 'min'),
            Max_Speed=('Average Speed', 'max'),
            Density=('Density', 'first'),
            Average_Flow=('Flow', 'mean'),
            Flow_Std_Dev=('Flow', 'std'),
            Min_Flow=('Flow', 'min'),
            Max_Flow=('Flow', 'max'),
            Simulation_Time=('Simulation Time (s)', 'first'),
            Time_Step=('Time Step (s)', 'first'),
            Animation_Interval=('Animation Interval (ms)', 'first'),
            Aggressive_Percent=('Aggressive %', 'first'),
            Normal_Percent=('Normal %', 'first'),
            Cautious_Percent=('Cautious %', 'first'),
            Polite_Percent=('Polite %', 'first'),
            Submissive_Percent=('Submissive %', 'first'),
        ).reset_index().rename(columns={
            'Average_Speed': 'Average Speed',
            'Variance': 'Variance of Average Speed',
            'Std_Dev': 'Standard Deviation of Average Speed',
            'Min_Speed': 'Minimum Average Speed',
            'Max_Speed': 'Maximum Average Speed',
            'Average_Flow': 'Average Flow',
            'Flow_Std_Dev': 'Standard Deviation of Flow',
            'Min_Flow': 'Minimum Flow',
            'Max_Flow': 'Maximum Flow',
            'Simulation_Time': 'Simulation Time (s)',
            'Time_Step': 'Time Step (s)',
            'Animation_Interval': 'Animation Interval (ms)',
            'Aggressive_Percent': 'Aggressive %',
            'Normal_Percent': 'Normal %',
            'Cautious_Percent': 'Cautious %',
            'Polite_Percent': 'Polite %',
            'Submissive_Percent': 'Submissive %',
        })
        
        # Save both dataframes to different sheets in the same Excel file
        excel_filename = os.path.join(folder_name, f'simulation_results.xlsx')
        
        # Use ExcelWriter to save multiple sheets to the same file
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df_detailed.to_excel(writer, sheet_name='Detailed Results', index=False)
            df_summary.to_excel(writer, sheet_name='Summary Results', index=False)
        
        print(f"\nSimulations complete!")
        print(f"Results saved to: {excel_filename}")
        print(f"  - Sheet 1: Detailed Results")
        print(f"  - Sheet 2: Summary Results")
        
        # Pass the folder name to the display function
        self.display_simulation_results(df_summary, folder_name)

    def display_simulation_results(self, df_summary, output_folder):
        """
        Display the simulation results in matplotlib figures showing:
        1. Average Speed vs Number of Vehicles
        2. Flow vs Density
        
        All output files are saved to the specified output folder.
        
        Args:
            df_summary: DataFrame containing summary statistics
            output_folder: Path to the folder where output files should be saved
        """
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Average Speed vs Number of Vehicles
        ax1.errorbar(
            df_summary['Number of Vehicles'], 
            df_summary['Average Speed'],
            yerr=df_summary['Standard Deviation of Average Speed'],
            fmt='o-', 
            ecolor='red',
            capsize=5,
            label='Average Speed with Std Dev'
        )
        
        # Add min/max as a shaded area
        ax1.fill_between(
            df_summary['Number of Vehicles'],
            df_summary['Minimum Average Speed'],
            df_summary['Maximum Average Speed'],
            alpha=0.2,
            color='blue',
            label='Min-Max Range'
        )
        
        # Add labels and legend for the first plot
        ax1.set_xlabel('Number of Vehicles')
        ax1.set_ylabel('Average Speed (m/s)')
        ax1.set_title('Impact of Vehicle Count on Average Speed')
        ax1.grid(True)
        ax1.legend()
        
        # Plot 2: Flow vs Density (Fundamental Diagram)
        ax2.errorbar(
            df_summary['Density'], 
            df_summary['Average Flow'],
            yerr=df_summary['Standard Deviation of Flow'],
            fmt='o-', 
            ecolor='green',
            capsize=5,
            label='Flow with Std Dev'
        )
        
        # Add min/max as a shaded area
        ax2.fill_between(
            df_summary['Density'],
            df_summary['Minimum Flow'],
            df_summary['Maximum Flow'],
            alpha=0.2,
            color='purple',
            label='Min-Max Range'
        )
        
        # Add labels and legend for the second plot
        ax2.set_xlabel('Density (vehicles/m)')
        ax2.set_ylabel('Flow (vehicles/s)')
        ax2.set_title('Fundamental Diagram: Flow vs Density')
        ax2.grid(True)
        ax2.legend()
        
        # Add a common title for both plots
        fig.suptitle(f'Traffic Simulation Results\n'
                    f'({self.num_simulations} simulations per vehicle count, '
                    f'{self.params["lanes_count"]} lanes, '
                    f'{self.params["distracted_percentage"]}% distracted drivers)',
                    fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust to make room for the suptitle
        
        # Save the combined figure
        combined_fig_path = os.path.join(output_folder, 'combined_plots.png')
        plt.savefig(combined_fig_path, dpi=300, bbox_inches='tight')
        print(f"Combined plots saved to: {combined_fig_path}")
        
        # Create and save individual plots as well
        
        # 1. Speed vs Vehicles plot
        fig_speed, ax_speed = plt.subplots(figsize=(10, 6))
        ax_speed.errorbar(
            df_summary['Number of Vehicles'], 
            df_summary['Average Speed'],
            yerr=df_summary['Standard Deviation of Average Speed'],
            fmt='o-', 
            ecolor='red',
            capsize=5,
            label='Average Speed with Std Dev'
        )
        ax_speed.fill_between(
            df_summary['Number of Vehicles'],
            df_summary['Minimum Average Speed'],
            df_summary['Maximum Average Speed'],
            alpha=0.2,
            color='blue',
            label='Min-Max Range'
        )
        ax_speed.set_xlabel('Number of Vehicles')
        ax_speed.set_ylabel('Average Speed (m/s)')
        ax_speed.set_title('Impact of Vehicle Count on Average Speed')
        ax_speed.grid(True)
        ax_speed.legend()
        speed_fig_path = os.path.join(output_folder, 'speed_vs_vehicles.png')
        fig_speed.savefig(speed_fig_path, dpi=300, bbox_inches='tight')
        print(f"Speed vs vehicles plot saved to: {speed_fig_path}")
        plt.close(fig_speed)
        
        # 2. Flow vs Density plot
        fig_flow, ax_flow = plt.subplots(figsize=(10, 6))
        ax_flow.errorbar(
            df_summary['Density'], 
            df_summary['Average Flow'],
            yerr=df_summary['Standard Deviation of Flow'],
            fmt='o-', 
            ecolor='green',
            capsize=5,
            label='Flow with Std Dev'
        )
        ax_flow.fill_between(
            df_summary['Density'],
            df_summary['Minimum Flow'],
            df_summary['Maximum Flow'],
            alpha=0.2,
            color='purple',
            label='Min-Max Range'
        )
        ax_flow.set_xlabel('Density (vehicles/m)')
        ax_flow.set_ylabel('Flow (vehicles/s)')
        ax_flow.set_title('Fundamental Diagram: Flow vs Density')
        ax_flow.grid(True)
        ax_flow.legend()
        flow_fig_path = os.path.join(output_folder, 'flow_vs_density.png')
        fig_flow.savefig(flow_fig_path, dpi=300, bbox_inches='tight')
        print(f"Flow vs density plot saved to: {flow_fig_path}")
        plt.close(fig_flow)
        
        # Also save the data as CSV for potential further analysis
        csv_path = os.path.join(output_folder, 'summary_data.csv')
        df_summary.to_csv(csv_path, index=False)
        print(f"Summary data CSV saved to: {csv_path}")
        
        # Create a simple README file with simulation parameters
        readme_path = os.path.join(output_folder, 'README.txt')
        with open(readme_path, 'w') as f:
            f.write(f"Traffic Simulation Results\n")
            f.write(f"========================\n\n")
            f.write(f"Date and Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Simulation Parameters:\n")
            f.write(f"  - Number of simulations per configuration: {self.num_simulations}\n")
            f.write(f"  - Road length: {self.params['road_length']} meters\n")
            f.write(f"  - Number of lanes: {self.params['lanes_count']}\n")
            f.write(f"  - Simulation time: {self.params['simulation_time']} seconds\n")
            f.write(f"  - Time step (dt): {self.params['dt']} seconds\n")
            f.write(f"  - Animation interval: {self.params['animation_interval']} ms\n")
            f.write(f"  - Vehicle counts tested: {', '.join(map(str, self.num_vehicles_array))}\n")
            f.write(f"  - Percentage of distracted drivers: {self.params['distracted_percentage']}%\n")
            f.write(f"  - Driver type distribution:\n")
            dist = self.params['driver_type_distribution']
            f.write(f"    * Aggressive: {dist[DriverType.AGGRESSIVE] * 100:.1f}%\n")
            f.write(f"    * Normal: {dist[DriverType.NORMAL] * 100:.1f}%\n")
            f.write(f"    * Cautious: {dist[DriverType.CAUTIOUS] * 100:.1f}%\n")
            f.write(f"    * Polite: {dist[DriverType.POLITE] * 100:.1f}%\n")
            f.write(f"    * Submissive: {dist[DriverType.SUBMISSIVE] * 100:.1f}%\n\n")
            f.write(f"Files in this folder:\n")
            f.write(f"  - combined_plots.png: Combined visualization of both key metrics\n")
            f.write(f"  - speed_vs_vehicles.png: Plot of average speed vs number of vehicles\n")
            f.write(f"  - flow_vs_density.png: Plot of traffic flow vs traffic density\n")
            f.write(f"  - simulation_results.xlsx: Detailed simulation results with two sheets\n")
            f.write(f"  - summary_data.csv: CSV version of the summary results\n")
        print(f"README file created at: {readme_path}")
        
        # Show the combined plot
        # plt.show()