import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Polygon, Circle
from matplotlib.widgets import Button, Slider, TextBox
import random
from collections import defaultdict
from enum import Enum
from vehicle import Vehicle
from trafficSimulation import TrafficSimulation

class SimulationGUI:
    def __init__(self):
        self.params = {
            'road_length': 1000,
            'lanes_count': 3,
            'n_vehicles': 30,
            'dt': 0.5,
            'simulation_time': 120,
            'animation_interval': 50
        }
        self.simulation = None
        self.fig = None
        
    def setup_start_screen(self):
        """Create and display the start screen with parameter controls."""
        self.fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(left=0.25, bottom=0.4)
        
        # Hide the axis
        ax.set_axis_off()
        
        # Add title
        ax.text(0.5, 0.85, 'Traffic Simulation', 
                fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)
        ax.text(0.5, 0.75, 'Set simulation parameters and press Start', 
                fontsize=14, ha='center', transform=ax.transAxes)
        
        # Create sliders for parameters
        ax_length = plt.axes([0.25, 0.6, 0.5, 0.03])
        ax_lanes = plt.axes([0.25, 0.55, 0.5, 0.03])
        ax_vehicles = plt.axes([0.25, 0.5, 0.5, 0.03])
        ax_simtime = plt.axes([0.25, 0.45, 0.5, 0.03])
        ax_dt = plt.axes([0.25, 0.4, 0.5, 0.03])
        ax_interval = plt.axes([0.25, 0.35, 0.5, 0.03])
        
        # Sliders
        self.slider_length = Slider(ax_length, 'Road Length (m)', 500, 2000, 
                                    valinit=self.params['road_length'], valstep=100)
        self.slider_lanes = Slider(ax_lanes, 'Lanes', 1, 5, 
                                  valinit=self.params['lanes_count'], valstep=1)
        self.slider_vehicles = Slider(ax_vehicles, 'Vehicles', 5, 50, 
                                     valinit=self.params['n_vehicles'], valstep=5)
        self.slider_simtime = Slider(ax_simtime, 'Sim Time (s)', 30, 300, 
                                    valinit=self.params['simulation_time'], valstep=30)
        self.slider_dt = Slider(ax_dt, 'Time Step (s)', 0.1, 1.0, 
                               valinit=self.params['dt'], valstep=0.1)
        self.slider_interval = Slider(ax_interval, 'Animation Speed', 10, 100, 
                                     valinit=self.params['animation_interval'], valstep=10)
        
        # Start button
        ax_start = plt.axes([0.4, 0.2, 0.2, 0.1])
        self.button_start = Button(ax_start, 'Start Simulation')
        self.button_start.on_clicked(self.start_simulation)
        
        # Connect update functions
        self.slider_length.on_changed(self.update_params)
        self.slider_lanes.on_changed(self.update_params)
        self.slider_vehicles.on_changed(self.update_params)
        self.slider_simtime.on_changed(self.update_params)
        self.slider_dt.on_changed(self.update_params)
        self.slider_interval.on_changed(self.update_params)
        
        plt.show()
        
    def update_params(self, val):
        """Update parameters when sliders change."""
        self.params['road_length'] = self.slider_length.val
        self.params['lanes_count'] = int(self.slider_lanes.val)
        self.params['n_vehicles'] = int(self.slider_vehicles.val)
        self.params['simulation_time'] = self.slider_simtime.val
        self.params['dt'] = self.slider_dt.val
        self.params['animation_interval'] = self.slider_interval.val
        
    def start_simulation(self, event):
        """Start the simulation with the selected parameters."""
        plt.close(self.fig)  # Close start screen
        
        # Create and run simulation with selected parameters
        self.simulation = TrafficSimulation(
            road_length=self.params['road_length'],
            lanes_count=self.params['lanes_count'],
            n_vehicles=self.params['n_vehicles'],
            dt=self.params['dt'],
            simulation_time=self.params['simulation_time'],
            animation_interval=self.params['animation_interval']
        )
        
        self.simulation.run_simulation()