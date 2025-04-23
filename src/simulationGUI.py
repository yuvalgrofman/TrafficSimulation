import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Polygon, Circle
from matplotlib.widgets import Button, Slider, TextBox, RadioButtons, CheckButtons
import random
from collections import defaultdict
from enum import Enum
from vehicle import Vehicle, DriverType
from trafficSimulation import TrafficSimulation

class SimulationGUI:
    def __init__(self):
        self.params = {
            'road_length': 1000,
            'lanes_count': 3,
            'n_vehicles': 30,
            'dt': 0.5,
            'simulation_time': 120,
            'animation_interval': 50,
            'distracted_percentage': 0  # Adding default percentage of distracted drivers
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
        
        # Text boxes
        ax_length = plt.axes([textbox_left, param_top - 0.015, textbox_width, param_height])
        ax_lanes = plt.axes([textbox_left, param_top - param_spacing - 0.015, textbox_width, param_height])
        ax_vehicles = plt.axes([textbox_left, param_top - 2*param_spacing - 0.015, textbox_width, param_height])
        ax_simtime = plt.axes([textbox_left, param_top - 3*param_spacing - 0.015, textbox_width, param_height])
        ax_dt = plt.axes([textbox_left, param_top - 4*param_spacing - 0.015, textbox_width, param_height])
        ax_interval = plt.axes([textbox_left, param_top - 5*param_spacing - 0.015, textbox_width, param_height])
        ax_distracted_percentage = plt.axes([textbox_left, param_top - 6*param_spacing - 0.015, textbox_width, param_height])
        
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
        ax_save_anim = plt.axes([0.35, 0.08, 0.1, 0.05])
        self.checkbox_save_anim = CheckButtons(ax_save_anim, ['Save Animation'], [False])
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
        
        # Connect update functions
        self.textbox_length.on_submit(self.update_params)
        self.textbox_lanes.on_submit(self.update_params)
        self.textbox_vehicles.on_submit(self.update_params)
        self.textbox_simtime.on_submit(self.update_params)
        self.textbox_dt.on_submit(self.update_params)
        self.textbox_interval.on_submit(self.update_params)
        self.textbox_distracted_percentage.on_submit(self.update_params)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
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
            self.params = {
                'road_length': 1000,
                'lanes_count': 3,
                'n_vehicles': 30,
                'dt': 0.5,
                'simulation_time': 120,
                'animation_interval': 50,
                'distracted_percentage': 0
            }
        
    def create_simulation(self):
        """Create a simulation instance with the current parameters."""
        # Create simulation with selected parameters
        simulation = TrafficSimulation(
            road_length=self.params['road_length'],
            lanes_count=self.params['lanes_count'],
            n_vehicles=self.params['n_vehicles'],
            dt=self.params['dt'],
            simulation_time=self.params['simulation_time'],
            animation_interval=self.params['animation_interval'],
            distracted_percentage=self.params['distracted_percentage']  # Pass distracted percentage
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
        self.simulation.debug = True
        
        # Run the simulation without animation for the specified number of steps
        self.simulation.run_without_animation(steps=self.non_animated_steps)