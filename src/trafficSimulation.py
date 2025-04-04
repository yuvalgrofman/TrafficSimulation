import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Polygon, Circle
from matplotlib.widgets import Button, Slider, TextBox
import random
from collections import defaultdict
from enum import Enum
from vehicle import Vehicle, DriverType

class TrafficSimulation:
    def __init__(self, road_length=1000, lanes_count=3, n_vehicles=30, dt=0.5, 
                 simulation_time=100, animation_interval=50):
        self.road_length = road_length  # length of the road (m)
        self.lanes_count = lanes_count  # number of lanes
        self.n_vehicles = n_vehicles  # number of vehicles
        self.dt = dt  # simulation time step (s)
        self.simulation_time = simulation_time  # total simulation time (s)
        self.animation_interval = animation_interval  # animation interval (ms)
        
        self.vehicles = []
        self.time = 0
        
        # Animation and control variables
        self.is_paused = False
        self.anim = None
        self.fig = None
        
        # Statistics tracking
        self.average_speeds = []
        self.lane_distributions = []
        self.lane_changes = 0
        
        # Initialize vehicles
        self.initialize_vehicles()
        
    def initialize_vehicles(self):
        """Initialize vehicles with random positions, velocities, and lanes."""
        # Clear existing vehicles if any
        self.vehicles = []
        
        for i in range(self.n_vehicles):
            # Random position (ensuring no overlaps)
            while True:
                position = random.uniform(0, self.road_length)
                lane = random.randint(0, self.lanes_count - 1)
                
                # Check for overlap with existing vehicles
                overlap = False
                for vehicle in self.vehicles:
                    if (vehicle.lane == lane and 
                        abs(vehicle.position - position) < max(vehicle.length, 10)):
                        overlap = True
                        break
                
                if not overlap:
                    break
            
            # Random desired velocity (m/s) - between 20 and 40 m/s (72-144 km/h)
            desired_velocity = random.uniform(20, 35)
            
            # Assign driver type with different probabilities
            driver_type = random.choices(
                list(DriverType),
                weights=[0.15, 0.4, 0.15, 0.15, 0.15]  # 15% aggressive, 40% normal, etc.
            )[0]
            
            # Set visualization dimensions
            vis_height, vis_width = 0.2, 20  # default dimensions
            
            # Create vehicle (starting at 70% of desired speed)
            vehicle = Vehicle(
                id=i,
                position=position,
                velocity=0.7 * desired_velocity,
                lane=lane,
                desired_velocity=desired_velocity,
                driver_type=driver_type,
                vis_height=vis_height,
                vis_width=vis_width
            )
            
            self.vehicles.append(vehicle)
            
    def run_step(self):
        """Run one simulation step."""
        if self.is_paused:
            return
            
        # Update all vehicles
        prev_lanes = {v.id: v.lane for v in self.vehicles}
        
        for vehicle in self.vehicles:
            vehicle.update(self.dt, self.vehicles, self.lanes_count, self.road_length)
            
        # Count lane changes
        for v in self.vehicles:
            if prev_lanes[v.id] != v.lane:
                self.lane_changes += 1
                
        # Record statistics
        average_speed = sum(v.velocity for v in self.vehicles) / len(self.vehicles)
        self.average_speeds.append(average_speed)
        
        lane_counts = defaultdict(int)
        for v in self.vehicles:
            lane_counts[v.lane] += 1
        self.lane_distributions.append(dict(lane_counts))
        
        self.time += self.dt
            
    def setup_animation(self):
        """Set up the animation."""
        # Create figure and axis
        fig = plt.figure(figsize=(14, 10))
        
        # Main road axis
        ax1 = plt.subplot2grid((5, 7), (0, 0), colspan=7, rowspan=3)
        # Speed graph axis
        ax2 = plt.subplot2grid((5, 7), (3, 0), colspan=7, rowspan=1)
        # Button controls
        ax_pause = plt.subplot2grid((5, 7), (4, 0), colspan=2, rowspan=1)
        ax_reset = plt.subplot2grid((5, 7), (4, 2), colspan=2, rowspan=1)
        
        self.fig = fig
        
        # Set axis limits for road
        ax1.set_xlim(0, self.road_length)
        ax1.set_ylim(-1, self.lanes_count)
        
        # Set axis labels and title
        ax1.set_xlabel('Position (m)')
        ax1.set_ylabel('Lane')
        ax1.set_title('Traffic Simulation with Different Driver Types')
        
        # Draw lane markings
        for i in range(self.lanes_count - 1):
            ax1.axhline(y=i + 0.5, color='white', linestyle='--')
            
        # Set road color
        ax1.set_facecolor('gray')
        
        # Remove ticks on y-axis and set custom lane labels
        ax1.set_yticks(np.arange(0, self.lanes_count))
        ax1.set_yticklabels([f'Lane {i+1}' for i in range(self.lanes_count)])
        
        # Set up the speed plot
        ax2.set_xlim(0, self.simulation_time)
        ax2.set_ylim(0, 100)  # Assuming max speed around 40 m/s
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Avg. Speed (km/h)')
        ax2.set_title('Average Traffic Speed')
        ax2.grid(True)
        
        # Line for average speed
        speed_line = ax2.plot([], [], 'r-', lw=2, label='km/h')
        ax2.legend()
        
        # Create a text element for statistics
        stats_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                              fontsize=10, va='top', ha='left')
        
        # Create legend for driver types
        legend_elements = [
            plt.Rectangle((0,0),1,1,fc=(0.8, 0.2, 0.2), label='Aggressive'),
            plt.Rectangle((0,0),1,1,fc=(0.2, 0.6, 0.2), label='Normal'),
            plt.Rectangle((0,0),1,1,fc=(0.2, 0.2, 0.8), label='Cautious'),
            plt.Rectangle((0,0),1,1,fc=(0.8, 0.8, 0.2), label='Polite'),
            plt.Rectangle((0,0),1,1,fc=(0.6, 0.2, 0.8), label='Submissive')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Create buttons
        self.button_pause = Button(ax_pause, 'Pause')
        self.button_pause.on_clicked(self.toggle_pause)
        
        self.button_reset = Button(ax_reset, 'Reset')
        self.button_reset.on_clicked(self.reset_simulation)
        
        # Return all elements that need to be updated
        return fig, ax1, ax2, speed_line, stats_text
    
    def toggle_pause(self, event):
        """Toggle pause/play state."""
        self.is_paused = not self.is_paused
        self.button_pause.label.set_text('Play' if self.is_paused else 'Pause')
        plt.draw()
    
    def reset_simulation(self, event):
        """Reset the simulation."""
        self.time = 0
        self.lane_changes = 0
        self.average_speeds = []
        self.lane_distributions = []
        self.initialize_vehicles()
        plt.draw()
        
    def draw_car(self, ax, vehicle):
        """Draw a car with its ID displayed."""
        x = vehicle.position
        y = vehicle.lane
        vis_length = vehicle.vis_width
        vis_height = vehicle.vis_height
        
        # Car body (main rectangle)
        body = Rectangle(
            (x - vis_length/2, y - vis_height/2),
            vis_length, vis_height,
            angle=0, color=vehicle.color, ec='black'
        )
        ax.add_patch(body)
        
        # Add vehicle ID text
        ax.text(x, y, str(vehicle.id), ha='center', va='center', 
                color='white', fontsize=8, fontweight='bold')
        
    def animate(self, frame):
        """Update animation for each frame."""
        # Run simulation for current frame
        self.run_step()
        
        # Clear previous vehicle patches
        ax = plt.gcf().axes[0]
        for patch in ax.patches:
            patch.remove()
        
        # Clear previous texts (except stats text)
        for txt in ax.texts[1:]:  # Skip the first text which is stats_text
            txt.remove()
            
        # Create new car representations
        for vehicle in self.vehicles:
            self.draw_car(ax, vehicle)
            
        # Update speed plot
        speed_line = plt.gcf().axes[1].get_lines()[0]
        
        times = np.arange(0, self.time, self.dt)
        kmh_speeds = [s * 3.6 for s in self.average_speeds]
        speed_line.set_data(times, kmh_speeds)
        
        # Update statistics text
        current_avg_speed = self.average_speeds[-1] if self.average_speeds else 0
        lane_counts = self.lane_distributions[-1] if self.lane_distributions else {}
        
        stats_info = (
            f"Time: {self.time:.1f}s\n"
            f"Avg Speed: {current_avg_speed:.1f} m/s ({current_avg_speed * 3.6:.1f} km/h)\n"
            f"Lane Changes: {self.lane_changes}\n"
            f"Vehicles per lane: {', '.join([f'Lane {k+1}: {v}' for k, v in sorted(lane_counts.items())])}"
        )
        
        stats_text = plt.gcf().axes[0].texts[0]
        stats_text.set_text(stats_info)
        
        # First text is stats_text, rest are vehicle IDs
        return [stats_text] + ax.patches + [speed_line] + ax.texts[1:]
    
    def run_simulation(self):
        """Run the full simulation with animation."""
        # Set up the animation
        fig, ax1, ax2, speed_line, stats_text = self.setup_animation()
        
        # Create animation
        self.anim = animation.FuncAnimation(
            fig, self.animate, 
            frames=int(self.simulation_time / self.dt),
            interval=self.animation_interval, 
            blit=True,
            cache_frame_data=False  # Fix for animation function
        )
        
        # Display animation
        plt.tight_layout()
        plt.show()
        
        return self.anim