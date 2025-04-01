import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from typing import List, Optional

class Car:
    def __init__(self, position: float, velocity: float, car_length: float = 5.0,
                 desired_speed: float = 120.0, time_headway: float = 1.5,
                 min_gap: float = 2.0, acceleration: float = 0.3,
                 deceleration: float = 2.0, delta: float = 4.0):
        """
        Initialize a car with IDM parameters.
        Speeds are in km/h, distances in m, accelerations in m/s^2.
        """
        self.position = position  # m
        self.velocity = velocity / 3.6  # Convert km/h to m/s
        self.acceleration = 0.0  # m/s^2
        
        # Car properties
        self.length = car_length  # m
        
        # IDM parameters
        self.desired_speed = desired_speed / 3.6  # Convert km/h to m/s
        self.time_headway = time_headway  # s
        self.min_gap = min_gap  # m
        self.max_acceleration = acceleration  # m/s^2
        self.comfortable_deceleration = deceleration  # m/s^2
        self.delta = delta  # acceleration exponent
        
    def update_acceleration(self, leading_car: Optional['Car'] = None):
        """Update acceleration based on IDM model."""
        # If no leading car, use virtual car at a fixed distance
        if leading_car is None:
            gap = 1000  # m
        else:
            gap = leading_car.position - self.position - leading_car.length
        
        # Calculate desired gap
        desired_gap = (self.min_gap + 
                      max(0, self.velocity * self.time_headway))
        
        # IDM formula for acceleration
        self.acceleration = self.max_acceleration * (
            1 - (self.velocity / self.desired_speed) ** self.delta - 
            (desired_gap / max(gap, 0.1)) ** 2
        )
    
    def update_position(self, dt: float):
        """Update position and velocity using ballistic model."""
        new_velocity = self.velocity + self.acceleration * dt
        # Ensure non-negative velocity
        self.velocity = max(0, new_velocity)
        
        # Ballistic update
        self.position += self.velocity * dt + 0.5 * self.acceleration * dt**2


class Simulation:
    def __init__(self, road_length: float = 240000, inflow_rate: float = 0.1,
                 min_speed: float = 80.0, max_speed: float = 120.0,
                 dt: float = 0.1):
        """
        Initialize traffic simulation.
        road_length: length of the road in meters
        inflow_rate: number of cars entering per second
        min/max_speed: range of speeds for new cars in km/h
        dt: simulation time step in seconds
        """
        self.road_length = road_length  # m
        self.inflow_rate = inflow_rate  # cars/second
        self.min_speed = min_speed  # km/h
        self.max_speed = max_speed  # km/h
        self.dt = dt  # s
        
        self.cars: List[Car] = []
        self.time = 0.0  # s
        self.next_inflow_time = 0.0  # s
        
    def add_car(self):
        """Add a new car at the start of the road."""
        # Random speed in the specified range
        speed = random.uniform(self.min_speed, self.max_speed)
        
        # Choose random driver type (affects time headway)
        driver_type = random.random()
        if driver_type < 0.2:  # Aggressive driver
            time_headway = random.uniform(0.8, 1.0)
        elif driver_type < 0.8:  # Normal driver
            time_headway = random.uniform(1.0, 1.5)
        else:  # Cautious driver
            time_headway = random.uniform(1.5, 2.0)
        
        new_car = Car(position=0.0, velocity=speed, time_headway=time_headway)
        
        # Insert car at the beginning of the list (cars are sorted by position in ascending order)
        self.cars.insert(0, new_car)
    
    def remove_cars_at_end(self):
        """Remove cars that have reached the end of the road."""
        self.cars = [car for car in self.cars if car.position < self.road_length]
    
    def update(self):
        """Update the simulation by one time step."""
        self.time += self.dt
        
        # Check if it's time to add a new car
        if self.time >= self.next_inflow_time:
            self.add_car()
            self.next_inflow_time = self.time + 1 / self.inflow_rate
        
        # Update accelerations for all cars
        for i, car in enumerate(self.cars):
            if i < len(self.cars) - 1:
                leading_car = self.cars[i + 1]
                car.update_acceleration(leading_car)
            else:
                car.update_acceleration(None)  # Leading car
        
        # Update positions for all cars
        for car in self.cars:
            car.update_position(self.dt)
        
        # Remove cars that have reached the end
        self.remove_cars_at_end()
    
    def get_car_positions(self):
        """Return the positions of all cars."""
        return [car.position for car in self.cars]
    
    def get_car_velocities(self):
        """Return the velocities of all cars in km/h."""
        return [car.velocity * 3.6 for car in self.cars]
    
    def get_car_accelerations(self):
        """Return the accelerations of all cars."""
        return [car.acceleration for car in self.cars]


class Visualization:
    def __init__(self, simulation: Simulation, update_interval: int = 50):
        """
        Initialize visualization for traffic simulation.
        update_interval: milliseconds between animation frames
        """
        self.simulation = simulation
        self.update_interval = update_interval
        
        # Create figure and axes
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.tight_layout(pad=3.0)
        
        # Initialize plots
        self.cars_scatter = self.ax1.scatter([], [], s=30, c='blue')
        self.ax1.set_xlim(0, simulation.road_length)
        self.ax1.set_ylim(-1, 1)
        self.ax1.set_xlabel('Position (m)')
        self.ax1.set_yticks([])
        self.ax1.set_title('Car Positions on Road')
        
        self.speed_scatter = self.ax2.scatter([], [], s=30, c=[])
        self.speed_cbar = self.fig.colorbar(self.speed_scatter, ax=self.ax2)
        self.speed_cbar.set_label('Speed (km/h)')
        self.ax2.set_xlim(0, simulation.road_length)
        self.ax2.set_ylim(-1, 1)
        self.ax2.set_xlabel('Position (m)')
        self.ax2.set_yticks([])
        self.ax2.set_title('Car Speeds')
        
        # Set animation
        self.animation = FuncAnimation(
            self.fig, self.update, interval=self.update_interval,
            blit=True, cache_frame_data=False
        )
    
    def update(self, frame):
        """Update the visualization for each animation frame."""
        # Update simulation
        self.simulation.update()
        
        # Get current data
        positions = self.simulation.get_car_positions()
        velocities = self.simulation.get_car_velocities()
        
        # Update position plot
        if positions:
            self.cars_scatter.set_offsets([(pos, 0) for pos in positions])
        else:
            self.cars_scatter.set_offsets(np.empty((0, 2)))
        
        # Update speed plot with color mapping
        if positions:
            self.speed_scatter.set_offsets([(pos, 0) for pos in positions])
            self.speed_scatter.set_array(np.array(velocities))
            self.speed_scatter.set_clim(self.simulation.min_speed, self.simulation.max_speed)
        else:
            self.speed_scatter.set_offsets(np.empty((0, 2)))
            self.speed_scatter.set_array(np.array([]))
        
        return self.cars_scatter, self.speed_scatter
    
    def show(self):
        """Display the animation."""
        plt.show()


def run_simulation(simulation_time: float = 3600, visualization: bool = True):
    """
    Run traffic simulation for the specified time.
    simulation_time: time to run in seconds
    visualization: whether to show visualization
    """
    sim = Simulation()
    
    if visualization:
        vis = Visualization(sim)
        vis.show()
    else:
        # Run simulation without visualization
        steps = int(simulation_time / sim.dt)
        for _ in range(steps):
            sim.update()
            
        # Final statistics
        velocities = sim.get_car_velocities()
        if velocities:
            print(f"Final number of cars: {len(velocities)}")
            print(f"Average speed: {np.mean(velocities):.2f} km/h")
            print(f"Min speed: {min(velocities):.2f} km/h")
            print(f"Max speed: {max(velocities):.2f} km/h")


if __name__ == "__main__":
    # Run simulation with visualization for 1 hour
    run_simulation(simulation_time=3600, visualization=True)