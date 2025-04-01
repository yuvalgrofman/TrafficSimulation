from typing import Optional
import numpy as np

# Global parameter for number of lanes
NUM_LANES = 3  # Default value, can be changed as needed

class Car:
    car_counter = 0  # Static counter to assign unique IDs

    def __init__(self, vehicle_type: str, driver_type: str, 
                 position: float, velocity: float, car_length: float = 5.0,
                 desired_speed: float = 120.0, time_headway: float = 1.5,
                 min_gap: float = 2.0, acceleration: float = 0.3, deceleration: float = 2.0, 
                 acceleration_exponent: float = 4.0, lane: int = 1,
                 safe_deceleration_limit: float = 2.0, change_threshold: float = 0.1, 
                 lane_change_bias: float = 0.1, politeness_factor: float = 0.2):
        self.id = f"{Car.car_counter:05d}"  # Assign unique 5-digit ID
        Car.car_counter += 1

        # Vehicle and driver types
        self.vehicle_type = vehicle_type
        self.driver_type = driver_type
        
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
        self.acceleration_exponent = acceleration_exponent  # acceleration exponent
        
        # New parameters
        self.safe_deceleration_limit = safe_deceleration_limit  # Safe deceleration limit
        self.change_threshold = change_threshold  # Changing threshold
        self.lane_change_bias = lane_change_bias  # Lane change bias
        self.politeness_factor = politeness_factor  # Politeness factor
        
        # Lane information
        assert 1 <= lane <= NUM_LANES, "Lane must be between 1 and NUM_LANES"
        self.lane = max(1, min(lane, NUM_LANES))  # Ensure lane is within valid range
    
    def update_acceleration(self, leading_car: Optional['Car'] = None):
        """Update acceleration based on IDM model."""
        if leading_car is None:
            gap = 1000  # m
            lead_velocity = self.velocity  # If no leading car, assume it's moving at the same speed
        else:
            gap = leading_car.position - self.position - leading_car.length
            lead_velocity = leading_car.velocity
        
        # Calculate acceleration using IDM formula
        self.acceleration = Car.compute_acceleration(self, gap, lead_velocity)
    
    def update_position(self, dt: float):
        """Update position and velocity using ballistic model."""
        new_velocity = self.velocity + self.acceleration * dt
        # Ensure non-negative velocity
        self.velocity = max(0, new_velocity)
        
        # Ballistic update
        self.position += self.velocity * dt + 0.5 * self.acceleration * dt**2

    def safe_lane_change(self, lane: list) -> bool:


        # Sort cars in the target lane by position
        lane = sorted(lane, key=lambda car: car.position)
        
        previous_car = None
        for car in reversed(lane):  # Iterate from back to front
            if car.position < self.position:
                previous_car = car
                break
        
        if previous_car is None:
            return True  # No car behind means it's safe to change
        
        # Compute the acceleration of previous_car as if it were in the same lane
        gap = self.position - (previous_car.position + previous_car.length)
        lead_velocity = self.velocity  # Assume self is the leading vehicle
        
        new_acceleration = Car.compute_acceleration(self, gap, lead_velocity)
        
        return new_acceleration >= -previous_car.safe_deceleration_limit

    def compute_lane_utility(self, old_lane: list['Car'], new_lane: list['Car']) -> float:
        # 1. Find the first car behind self in the new lane (p_car)
        p_car = None
        for car in new_lane:
            if car.position < self.position:
                if p_car is None or car.position > p_car.position:
                    p_car = car
        
        # 2. Calculate acceleration difference for p_car
        acceleration_difference_new_lane = 0
        if p_car is not None:
            # Calculate gap between p_car and self
            gap = self.position - p_car.position - p_car.length
            
            # 2.a. Calculate p_car's current acceleration (without self in the lane)
            current_acceleration = p_car.acceleration
            
            # 2.b. Calculate p_car's new acceleration (with self in the lane)
            front_gap = self.position - p_car.position - self.length
            front_velocity = self.velocity
            current_acceleration = Car.compute_acceleration(p_car, front_gap, front_velocity)
            
            # Calculate p_car's acceleration with self in the lane
            new_acceleration = Car.compute_acceleration(p_car, gap, self.velocity)
            
            # Calculate the acceleration difference
            acceleration_difference_new_lane = new_acceleration - current_acceleration
        
        # 3. Find the first car behind self in the current lane (p_car_current)
        p_car_current = None
        for car in old_lane:
            if car.position < self.position and car is not self:
                if p_car_current is None or car.position > p_car_current.position:
                    p_car_current = car
        
        # Calculate acceleration difference for p_car_current
        acceleration_difference_current_lane = 0
        if p_car_current is not None:
            # 3.a. Calculate p_car_current's current acceleration (with self in the lane)
            current_acceleration = p_car_current.acceleration
            
            # 3.b. Calculate p_car_current's new acceleration (without self in the lane)
            front_gap = self.position - p_car_current.position - self.length
            front_velocity = self.velocity
            new_acceleration = Car.compute_acceleration(p_car_current, front_gap, front_velocity)
            acceleration_difference_current_lane = new_acceleration - current_acceleration
        
        # 4. Find the first car in front of self in the new lane (f_car)
        f_car = None
        for car in new_lane:
            if car.position > self.position:
                if f_car is None or car.position < f_car.position:
                    f_car = car
        
        # 5.a. Calculate self's current acceleration
        current_acceleration = self.acceleration
        
        # 5.b. Calculate self's new acceleration in the new lane
        new_gap = 1000
        new_front_velocity = self.velocity
        if f_car is not None:
            new_gap = f_car.position - self.position - f_car.length
            new_front_velocity = f_car.velocity

        new_acceleration = Car.compute_acceleration(self, new_gap, new_front_velocity)
        
        acceleration_difference_self = new_acceleration - current_acceleration
        
        # 6. Compute the utility
        utility = acceleration_difference_self + self.politeness_factor * (
            acceleration_difference_new_lane + acceleration_difference_current_lane
        )
        
        return utility

    @staticmethod
    def compute_acceleration(back: 'Car', gap : int, front_velocity : int) -> float:
        # Calculate the relative velocity
        relative_velocity = back.velocity - front_velocity
        
        # Calculate the desired gap according to IDM
        desired_gap = (back.min_gap + 
                       back.velocity * back.time_headway + 
                       (back.velocity * relative_velocity) / 
                       (2 * np.sqrt(back.max_acceleration * back.comfortable_deceleration)))
        
        # Calculate acceleration using IDM formula
        acceleration = back.max_acceleration * (
            1 - (back.velocity / back.desired_speed) ** back.acceleration_exponent - 
            (desired_gap / max(gap, 0.1)) ** 2
        )
        
        return acceleration


class CarFactory:
    @staticmethod
    def create_vehicle(vehicle_type: str, driver_type: str, position: float, velocity: float, 
                      lane: int = 1, **kwargs) -> Car:
        # Base parameters for a regular car
        params = {
            'vehicle_type': vehicle_type,
            'driver_type': driver_type,
            'car_length': 5.0,
            'desired_speed': 120.0,
            'time_headway': 1.5,
            'min_gap': 2.0,
            'acceleration': 0.3,
            'deceleration': 2.0,
            'acceleration_exponent': 2.0,
            'lane': lane,
            'safe_deceleration_limit': 2.0,
            'change_threshold': 0.5,
            'lane_change_bias': 0.1,
            'politeness_factor': 0.2
        }

        # Update parameters based on vehicle type
        if vehicle_type.lower() == 'truck':
            params.update({
                'car_length': 10.0,
                'desired_speed': 90.0,
                'time_headway': 1.7,
                'min_gap': 3.0,
                'acceleration': 0.25,
                'deceleration': 2.0
            })
        elif vehicle_type.lower() == 'motorcycle':
            params.update({
                'car_length': 2.5,
                'desired_speed': 130.0,
                'time_headway': 1.0,
                'min_gap': 1.5,
                'acceleration': 0.4,
                'deceleration': 1.5
            })

        # Adjust for driver type
        driver_params = {}
        if driver_type.lower() == 'aggressive':
            driver_params = {
                'desired_speed': 140.0,
                'time_headway': 1.0,
                'min_gap': 1.5,
                'acceleration': 0.4,
                'deceleration': 3.0,
                'politeness_factor': 0.1
            }
        elif driver_type.lower() == 'cautious':
            driver_params = {
                'desired_speed': 100.0,
                'time_headway': 2.0,
                'min_gap': 3.0,
                'acceleration': 0.2,
                'deceleration': 1.5,
                'politeness_factor': 0.3
            }

        params.update(driver_params)
        params.update(kwargs)  # Override with any additional parameters

        return Car(position=position, velocity=velocity, **params)