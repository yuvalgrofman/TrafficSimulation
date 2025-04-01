import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from typing import List, Optional, Dict
from cars import Car, CarFactory, NUM_LANES

class Simulation:
    def __init__(self, road_length: float = 4000, inflow_rate: float = 0.1,
                 min_speed: float = 60.0, max_speed: float = 120.0,
                 dt: float = 0.1, num_lanes: int = NUM_LANES,
                 lane_changing: bool = False, artificial_cars: List[Car] = None):
        self.road_length = road_length  # m
        self.inflow_rate = inflow_rate  # cars/second/lane
        self.min_speed = min_speed  # km/h
        self.max_speed = max_speed  # km/h
        self.dt = dt  # s
        self.num_lanes = num_lanes
        self.lane_changing = lane_changing
        
        # Dictionary to store cars by lane (1 to num_lanes)
        self.cars: Dict[int, List[Car]] = {lane: [] for lane in range(1, num_lanes + 1)}
        self.time = 0.0  # s
        self.next_inflow_time = artificial_cars[0]["inflow_time"] if artificial_cars else 0.0

        # Artificial cars
        self.artificial_cars = artificial_cars if artificial_cars else None
        self.artificial_car_index = 0
    
    def add_car(self, lane: int, car: Optional[Car] = None, ):
        """Add a car to the specified lane."""
        vehicle_type = random.choice(["car", "truck", "motorcycle"])
        driver_type = random.choice(["aggressive", "normal", "cautious"])
        velocity = random.uniform(self.min_speed, self.max_speed)

        car_to_add = car if car else CarFactory.create_vehicle(
                vehicle_type=vehicle_type,
                driver_type=driver_type,
                position=0.0,
                velocity=velocity,
                lane=lane)
        
        # Insert car at the beginning of the lane's car list
        self.cars[lane].insert(lane, car_to_add)
        
    def add_to_lanes(self):
        """Add new cars to lanes according to inflow rate."""
        for lane in range(1, self.num_lanes + 1):
            self.add_car(lane)
    
    def remove_cars_at_end(self):
        """Remove cars that have reached the end of the road in any lane."""
        for lane in self.cars:
            self.cars[lane] = [car for car in self.cars[lane] 
                              if car.position < self.road_length]
    
    def find_leading_car(self, car: Car) -> Optional[Car]:
        lane_cars = self.cars[car.lane]
        try:
            car_index = lane_cars.index(car)
            if car_index < len(lane_cars) - 1:
                return lane_cars[car_index + 1]
        except ValueError:
            pass
        return None

    def handle_lane_changes(self):
        """Process lane changes for all cars in the simulation."""
        if not self.lane_changing:
            return
            
        # Process each lane
        for lane in range(1, self.num_lanes + 1):
            # We need to create a copy of the list since we might be modifying it
            cars_in_lane = self.cars[lane].copy()
            
            for car in cars_in_lane:
                # Skip cars that have already changed lanes in this time step
                if hasattr(car, 'changed_lane_this_step') and car.changed_lane_this_step:
                    continue
                    
                best_lane = lane
                best_utility = car.compute_lane_utility(self.cars[lane], self.cars[lane])  # Current lane utility
                
                # Check left lane if not in leftmost lane
                if lane > 1:
                    left_lane = lane - 1
                    if car.safe_lane_change(self.cars[left_lane]):
                        left_utility = car.compute_lane_utility(self.cars[lane], self.cars[left_lane])
                        if left_utility > best_utility and left_utility > (car.change_threshold + car.lane_change_bias):
                            best_lane = left_lane
                            best_utility = left_utility
                
                # Check right lane if not in rightmost lane
                if lane < self.num_lanes:
                    right_lane = lane + 1
                    if car.safe_lane_change(self.cars[right_lane]):
                        right_utility = car.compute_lane_utility(self.cars[lane], self.cars[right_lane])
                        if right_utility > best_utility and right_utility > (car.change_threshold + car.lane_change_bias):
                            best_lane = right_lane
                            best_utility = right_utility
                
                # If we found a better lane, change to it
                if best_lane != lane:
                    # Remove from current lane
                    self.cars[lane].remove(car)
                    
                    # Add to new lane (maintain sorted order by position)
                    car.lane = best_lane
                    
                    # Find the proper insertion point in the new lane (sort by position ascending)
                    insertion_index = 0
                    for i, other_car in enumerate(self.cars[best_lane]):
                        if car.position <= other_car.position:
                            insertion_index = i
                            break
                        insertion_index = i + 1
                    
                    self.cars[best_lane].insert(insertion_index, car)
                    
                    # Mark that this car has changed lanes
                    car.changed_lane_this_step = True
        
        # Reset the "changed_lane_this_step" flag for all cars
        for lane in self.cars:
            for car in self.cars[lane]:
                car.changed_lane_this_step = False

    def artificial_update(self):
        self.time += self.dt
        
        # Check if it's time to add new cars
        if self.time >= self.next_inflow_time:
            i = self.artificial_car_index
            self.add_car(self.artificial_cars[i]["car"].lane, self.artificial_cars[i]["car"])

            if i + 1 < len(self.artificial_cars):
                self.next_inflow_time += self.artificial_cars[i + 1]["inflow_time"]
                self.artificial_car_index += 1
        
        # Update accelerations for all cars in all lanes
        for lane in self.cars:
            for i, car in enumerate(self.cars[lane]):
                leading_car = self.find_leading_car(car)
                car.update_acceleration(leading_car)
        
        # Process lane changes if enabled
        if self.lane_changing:
            self.handle_lane_changes()
        
        # Update positions for all cars in all lanes
        for lane in self.cars:
            for car in self.cars[lane]:
                car.update_position(self.dt)
        
        # Remove cars that have reached the end
        self.remove_cars_at_end()
    
    def update(self):
        """Update the simulation by one time step."""
        if self.artificial_cars:
            self.artificial_update()
            return
            
        self.time += self.dt
        
        # Check if it's time to add new cars
        if self.time >= self.next_inflow_time:
            self.add_to_lanes()
            self.next_inflow_time += 1.0 / self.inflow_rate
        
        # Update accelerations for all cars in all lanes
        for lane in self.cars:
            for i, car in enumerate(self.cars[lane]):
                leading_car = self.find_leading_car(car)
                car.update_acceleration(leading_car)
        
        # Process lane changes if enabled
        if self.lane_changing:
            self.handle_lane_changes()
        
        # Update positions for all cars in all lanes
        for lane in self.cars:
            for car in self.cars[lane]:
                car.update_position(self.dt)
        
        # Remove cars that have reached the end
        self.remove_cars_at_end()
    
    
    def get_car_data(self):
        """Return dictionaries of car positions, velocities, and lanes."""
        return self.cars
        