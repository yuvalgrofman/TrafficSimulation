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
                 simulation_time=100, animation_interval=50, distracted_percentage=10, to_print=False,
                 driver_distribution=None):
        self.road_length = road_length  # length of the road (m)
        self.lanes_count = lanes_count  # number of lanes
        self.n_vehicles = n_vehicles  # number of vehicles
        self.dt = dt  # simulation time step (s)
        self.simulation_time = simulation_time  # total simulation time (s)
        self.animation_interval = animation_interval  # animation interval (ms)
        
        self.vehicles = []
        self.time = 0
        self.to_print = to_print  # Flag to print vehicle information
        
        # Vehicle deployment schedule
        self.scheduled_vehicles = []
        
        # Animation and control variables
        self.is_paused = False
        self.fast_forward = False
        self.anim = None
        self.fig = None
        
        # Statistics tracking
        self.average_speeds = []
        self.lane_distributions = []
        self.lane_changes = 0
        
        # Obstacles list
        self.obstacles = []

        # Percentage of vehicles that can be distracted
        self.distracted_percentage = distracted_percentage
        
        # Debug flag
        self.debug = False

        # Driver distribution
        self.driver_distribution = driver_distribution if driver_distribution else {
            DriverType.AGGRESSIVE: 0.3,
            DriverType.NORMAL: 0.3,
            DriverType.CAUTIOUS: 0.2,
            DriverType.POLITE: 0.1,
            DriverType.SUBMISSIVE: 0.1
        }

        self.num_each_driver_type = {
            DriverType.AGGRESSIVE: int(self.driver_distribution[DriverType.AGGRESSIVE] * n_vehicles),
            DriverType.NORMAL: int(self.driver_distribution[DriverType.NORMAL] * n_vehicles),
            DriverType.CAUTIOUS: int(self.driver_distribution[DriverType.CAUTIOUS] * n_vehicles),
            DriverType.POLITE: int(self.driver_distribution[DriverType.POLITE] * n_vehicles),
            DriverType.SUBMISSIVE: int(self.driver_distribution[DriverType.SUBMISSIVE] * n_vehicles)
        }

        if sum(self.num_each_driver_type.values()) != n_vehicles:
            self.num_each_driver_type[DriverType.NORMAL] += n_vehicles - sum(self.num_each_driver_type.values())

        self.driver_types = []
        for driver_type, count in self.num_each_driver_type.items():
            # Append driver types based on their counts
            for _ in range(int(count)):
                self.driver_types.append(driver_type)
        
        # Reorder the driver types randomly
        random.shuffle(self.driver_types)
        
        # Initialize vehicles
        if n_vehicles > 0:
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
                
                # Check for overlap with obstacles
                for obstacle in self.obstacles:
                    if (obstacle['lane'] == lane and 
                        abs(obstacle['position'] - position) < max(20, 10)):
                        overlap = True
                        break
                
                if not overlap:
                    break
            
            # Random desired velocity (m/s) - between 20 and 40 m/s (72-126 km/h)
            desired_velocity = random.uniform(25, 35)
            
            # Assign driver type with different probabilities
            driver_type = self.driver_types[i]
            
            # Set visualization dimensions
            vis_height, vis_width = 0.2, 20  # default dimensions
            
            # Determine if this vehicle can be distracted based on distracted_percentage
            can_be_distracted = random.randint(1, 100) <= self.distracted_percentage
            
            # Create vehicle (starting at 70% of desired speed)
            vehicle = Vehicle(
                id=i,
                position=position,
                velocity=0.7 * desired_velocity,
                lane=lane,
                desired_velocity=desired_velocity,
                driver_type=driver_type,
                vis_height=vis_height,
                vis_width=vis_width,
                can_be_distracted=can_be_distracted  # Set distraction capability
            )
            
            self.vehicles.append(vehicle)

    def add_obstacle(self, position, lane):
        """Add a static obstacle to the simulation."""
        obstacle = {
            'position': position,
            'lane': lane,
            'width': 20,  # visual width of obstacle
            'height': 0.2  # visual height of obstacle
        }
        self.obstacles.append(obstacle)

    def deploy_scheduled_vehicle(self):
        """Check if any vehicles need to be deployed at the current time."""
        for i, vehicle_info in enumerate(self.scheduled_vehicles[:]):
            if vehicle_info['deployment_time'] <= self.time:
                # Find a suitable position
                position = vehicle_info.get('initial_position', 0)  # Use specified position or default to 0
                lane = vehicle_info['lane']
                
                # Check for overlap with existing vehicles
                overlap = True
                attempts = 0
                while overlap and attempts < 5:
                    overlap = False
                    for vehicle in self.vehicles:
                        if (vehicle.lane == lane and 
                            abs(vehicle.position - position) < max(vehicle.length, 20)):
                            overlap = True
                            position += 25  # Move further down the road
                            break
                    
                    # Check for overlap with obstacles
                    for obstacle in self.obstacles:
                        if (obstacle['lane'] == lane and 
                            abs(obstacle['position'] - position) < 20):
                            overlap = True
                            position += 25  # Move further down the road
                            break
                    
                    # If we've reached the end of the road, try a different lane
                    if position >= self.road_length:
                        position = 0
                        lane = (lane + 1) % self.lanes_count
                    
                    attempts += 1
                
                # If after multiple attempts we still have overlap, skip this vehicle
                if overlap:
                    print(f"Warning: Could not deploy vehicle at time {self.time}. Skipping.")
                    self.scheduled_vehicles.pop(i)
                    continue
                
                # Create new vehicle
                new_vehicle = Vehicle(
                    id=len(self.vehicles),
                    position=position,
                    velocity=0.7 * vehicle_info['desired_velocity'],
                    lane=lane,
                    desired_velocity=vehicle_info['desired_velocity'],
                    driver_type=vehicle_info['driver_type'],
                    vis_height=0.2,
                    vis_width=20,
                    can_be_distracted=vehicle_info['is_distracted'],
                )

                new_vehicle.set_driver_parameters()
                
                self.vehicles.append(new_vehicle)
                
                # Remove from scheduled list
                self.scheduled_vehicles.pop(i)
                break  # Only deploy one vehicle per time step to avoid conflicts
            
    def run_step(self):
        """Run one simulation step."""
        if self.is_paused:
            return
            
        # Check if any vehicles need to be deployed
        self.deploy_scheduled_vehicle()
        
        # Update all vehicles
        prev_lanes = {v.id: v.lane for v in self.vehicles}
        
        for vehicle in self.vehicles:
            # Update each vehicle, also passing obstacles information
            vehicle.update(self.dt, self.vehicles, self.lanes_count, self.road_length, current_time=self.time)
            
        # Count lane changes
        for v in self.vehicles:
            if prev_lanes[v.id] != v.lane:
                self.lane_changes += 1
                
        # Record statistics
        if self.vehicles:  # Only calculate if there are vehicles
            average_speed = sum(v.velocity for v in self.vehicles) / len(self.vehicles)
            self.average_speeds.append(average_speed)
        else:
            self.average_speeds.append(0)
        
        lane_counts = defaultdict(int)
        for v in self.vehicles:
            lane_counts[v.lane] += 1
        self.lane_distributions.append(dict(lane_counts))
        
        self.time += self.dt
        
        # Check simulation integrity
        if self.debug:
            self.check_simulation_integrity()
            
    def check_simulation_integrity(self):
        """Check for simulation problems like vehicle overlaps."""
        # Check for vehicle-vehicle overlaps
        for i, v1 in enumerate(self.vehicles):
            for v2 in self.vehicles[i+1:]:
                if v1.lane == v2.lane:
                    distance = abs(v1.position - v2.position)
                    if distance < (v1.vis_width/2 + v2.vis_width/2) * 0.8:  # 80% of combined widths
                        print(f"WARNING: Vehicles {v1.id} and {v2.id} overlapping in lane {v1.lane}!")
                        print(f"  V{v1.id} at {v1.position:.1f}, V{v2.id} at {v2.position:.1f}, Distance: {distance:.1f}")
        
            # Check for vehicle-obstacle overlaps
            for obs in self.obstacles:
                if v1.lane == obs['lane']:
                    distance = abs(v1.position - obs['position'])
                    if distance < (v1.vis_width/2 + obs['width']/2) * 0.8:  # 80% of combined widths
                        print(f"WARNING: Vehicle {v1.id} overlapping with obstacle at position {obs['position']} in lane {obs['lane']}!")
                        print(f"  V{v1.id} at {v1.position:.1f}, Obstacle at {obs['position']}, Distance: {distance:.1f}")
    
    def run_without_animation(self, steps=10):
        """Run simulation for specified steps without animation"""
        if self.to_print:
            print(f"Running {steps} steps without animation...")
            
            for i in range(steps):
                self.run_step()
                # Print debug info after each step
                print(f"\nStep {i+1}, Time: {self.time:.1f}")
                
                if self.vehicles:
                    avg_speed = sum(v.velocity for v in self.vehicles) / len(self.vehicles)
                    print(f"Average speed: {avg_speed:.1f} m/s ({avg_speed*3.6:.1f} km/h)")
                    print(f"Lane changes so far: {self.lane_changes}")
                
                # Print details for each vehicle
                print("Vehicle details:")
                for v in self.vehicles:
                    print(f"  Vehicle {v.id}: Lane {v.lane}, Pos {v.position:.1f}, Speed {v.velocity:.1f} m/s, Type {v.driver_type}")
                
                # Check for problems
                if self.debug:
                    self.check_simulation_integrity()
            
            print("Non-animated simulation complete")
            
            # Return average speed
            if self.vehicles:
                avg_speed = sum(v.velocity for v in self.vehicles) / len(self.vehicles)
                return avg_speed
            return -1
        else:
            for i in range(steps):
                self.run_step()
            
            print("Non-animated simulation complete")
            
            # Return average speed
            if self.vehicles:
                avg_speed = sum(v.velocity for v in self.vehicles) / len(self.vehicles)
                return avg_speed
            return -1
            
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
        ax1.axhline(y=-0.5, color='white', linestyle='--')
        for i in range(self.lanes_count - 1):
            ax1.axhline(y=i + 0.5, color='white', linestyle='--')
        ax1.axhline(y=self.lanes_count - 0.5, color='white', linestyle='--')

        # Set road color
        ax1.set_facecolor('gray')

        # Remove ticks on y-axis and set custom lane labels
        ax1.set_yticks(np.arange(0, self.lanes_count))
        ax1.set_yticklabels([f'Lane {i+1}' for i in range(self.lanes_count)])

        # Set up the speed plot
        ax2.set_xlim(0, self.simulation_time)
        ax2.set_ylim(0, 130)  # Assuming max speed around 40 m/s
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
        
        # Create a text element for scheduled vehicles info
        scheduled_text = ax1.text(0.98, 0.95, '', transform=ax1.transAxes,
                                fontsize=10, va='top', ha='right')

        # Create legend for driver types
        legend_elements = [
            plt.Rectangle((0,0),1,1,fc=(0.8, 0.2, 0.2), label='Aggressive'),
            plt.Rectangle((0,0),1,1,fc=(0.2, 0.6, 0.2), label='Normal'),
            plt.Rectangle((0,0),1,1,fc=(0.2, 0.2, 0.8), label='Cautious'),
            plt.Rectangle((0,0),1,1,fc=(0.8, 0.8, 0.2), label='Polite'),
            plt.Rectangle((0,0),1,1,fc=(0.6, 0.2, 0.8), label='Submissive'),
            plt.Rectangle((0,0),1,1,fc=(0, 0, 0), label='Obstacle')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')

        # Create buttons
        self.button_pause = Button(ax_pause, 'Pause')
        self.button_pause.on_clicked(self.toggle_pause)

        self.button_reset = Button(ax_reset, 'Reset')
        self.button_reset.on_clicked(self.reset_simulation)

        # Add keyboard event handler
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Return all elements that need to be updated
        return fig, ax1, ax2, speed_line[0], stats_text, scheduled_text

    def on_key_press(self, event):
        """Handle keyboard events for debugging."""
        if event.key == 'd':
            # Print driver information
            self.print_drivers_info()
        elif event.key == 'p':
            # Pause simulation
            self.is_paused = True
            self.button_pause.label.set_text('Play')
            plt.draw()
            print("Simulation paused")
        elif event.key == 'l':
            # Step forward once
            old_pause_state = self.is_paused
            self.is_paused = False
            self.run_step()
            self.is_paused = old_pause_state
            self.fig.canvas.draw_idle()
            print(f"Stepped forward to time {self.time:.1f}")
        elif event.key == 'r':
            # Resume simulation
            self.is_paused = False
            self.button_pause.label.set_text('Pause')
            plt.draw()
            print("Simulation resumed")
        elif event.key == '0':  # Number zero
            # Reset simulation
            self.reset_simulation(None)
            print("Simulation reset")
        elif event.key == 'q':
            # Quit simulation
            plt.close(self.fig)
            print("Simulation closed")
        elif event.key == 'x':
            # Fast forward mode toggle
            self.fast_forward = not self.fast_forward
            if self.fast_forward:
                self.animation_interval = 10  # Much faster
                print("Fast forward mode enabled")
            else:
                self.animation_interval = 50  # Back to normal
                print("Fast forward mode disabled")
            
            # Update animation interval if it exists
            if hasattr(self, 'anim') and self.anim:
                self.anim.event_source.interval = self.animation_interval
    
    def print_drivers_info(self):
        """Print detailed information about all drivers."""
        print("\n=== DRIVERS INFORMATION ===")
        print(f"Time: {self.time:.1f}s, Total vehicles: {len(self.vehicles)}")
        
        if not self.vehicles:
            print("No vehicles in simulation.")
            return
            
        # Sort by position for better readability
        sorted_vehicles = sorted(self.vehicles, key=lambda v: v.position)
        
        for v in sorted_vehicles:
            print(f"Vehicle {v.id}: Type {v.driver_type.name}, Lane {v.lane}, "
                  f"Pos {v.position:.1f}m, Speed {v.velocity:.1f}m/s ({v.velocity*3.6:.1f}km/h), "
                  f"Target {v.desired_velocity:.1f}m/s", 
                    f"Distraction: {'Yes' if v.can_be_distracted else 'No'}")
            
        # Print obstacle information
        if self.obstacles:
            print("\n=== OBSTACLES ===")
            for i, obs in enumerate(self.obstacles):
                print(f"Obstacle {i}: Lane {obs['lane']}, Position {obs['position']}m")
    
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
        self.vehicles = []
        self.is_paused = False
        self.fast_forward = False
        
        # Reset original scheduled vehicles
        self.scheduled_vehicles = list(self.original_scheduled_vehicles)
        
        if self.n_vehicles > 0:
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
    
    def draw_obstacle(self, ax, obstacle):
        """Draw an obstacle on the road."""
        x = obstacle['position']
        y = obstacle['lane']
        width = obstacle['width']
        height = obstacle['height']
        
        # Obstacle rectangle (black)
        rect = Rectangle(
            (x - width/2, y - height/2),
            width, height,
            angle=0, color='black', ec='red'
        )
        ax.add_patch(rect)
        
        # Add "X" text
        ax.text(x, y, "X", ha='center', va='center', 
                color='red', fontsize=10, fontweight='bold')
        
    def animate(self, frame):
        """Update animation for each frame."""
        # Run simulation for current frame
        self.run_step()
        
        # Clear previous vehicle patches
        ax = plt.gcf().axes[0]
        for patch in ax.patches:
            patch.remove()
        
        # Clear previous texts (except stats text and scheduled text)
        for txt in ax.texts[2:]:  # Skip the first 2 texts which are stats_text and scheduled_text
            txt.remove()
            
        # Create new car representations
        for vehicle in self.vehicles:
            self.draw_car(ax, vehicle)
        
        # Draw obstacles
        for obstacle in self.obstacles:
            self.draw_obstacle(ax, obstacle)
            
        # Update speed plot
        speed_line = plt.gcf().axes[1].get_lines()[0]
        
        # Fix: Create time array that matches the length of average_speeds
        times = np.linspace(0, self.time, len(self.average_speeds))
        kmh_speeds = [s * 3.6 for s in self.average_speeds]
        speed_line.set_data(times, kmh_speeds)
        
        # Update statistics text
        current_avg_speed = self.average_speeds[-1] if self.average_speeds else 0
        lane_counts = self.lane_distributions[-1] if self.lane_distributions else {}
        
        stats_info = (
            f"Time: {self.time:.1f}s\n"
            f"Vehicles: {len(self.vehicles)}\n"
            f"Avg Speed: {current_avg_speed:.1f} m/s ({current_avg_speed * 3.6:.1f} km/h)\n"
            f"Lane Changes: {self.lane_changes}\n"
            f"Vehicles per lane: {', '.join([f'Lane {k+1}: {v}' for k, v in sorted(lane_counts.items())])}"
        )
        
        stats_text = plt.gcf().axes[0].texts[0]
        stats_text.set_text(stats_info)
        
        # Update scheduled vehicles text
        scheduled_text = plt.gcf().axes[0].texts[1]
        next_scheduled = sorted(self.scheduled_vehicles, key=lambda x: x['deployment_time'])
        
        if next_scheduled:
            next_vehicle = next_scheduled[0]
            scheduled_info = (
                f"Next vehicle deployment:\n"
                f"Time: {next_vehicle['deployment_time']}s\n"
                f"Lane: {next_vehicle['lane'] + 1}\n"
                f"Scheduled: {len(self.scheduled_vehicles)} remaining"
            )
        else:
            scheduled_info = "No vehicles scheduled"
            
        scheduled_text.set_text(scheduled_info)
        
        # First texts are stats_text and scheduled_text, rest are vehicle IDs
        return [stats_text, scheduled_text] + ax.patches + [speed_line] + ax.texts[2:]
    
    
    def run_simulation(self, save_animation=False):
        """Run the full simulation with animation."""
        # Store original scheduled vehicles for reset
        self.original_scheduled_vehicles = list(self.scheduled_vehicles)
        
        # Set up the animation
        fig, ax1, ax2, speed_line, stats_text, scheduled_text = self.setup_animation()
        
        # Create animation
        self.anim = animation.FuncAnimation(
            fig, self.animate, 
            frames=int(self.simulation_time / self.dt),
            interval=self.animation_interval, 
            blit=True,
            cache_frame_data=False  # Fix for animation function
        )
        
        # Save animation if requested
        if save_animation:
            print("Saving animation to traffic_simulation.mp4...")
            writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Traffic Simulator'), 
                                          bitrate=1800)
            self.anim.save('traffic_simulation.mp4', writer=writer)
            print("Animation saved successfully.")
        
        # Display animation
        plt.tight_layout()
        plt.show()
        
        return self.anim