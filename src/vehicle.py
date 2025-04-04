import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Polygon, Circle
from matplotlib.widgets import Button, Slider, TextBox
import random
from collections import defaultdict
from enum import Enum

class DriverType(Enum):
    AGGRESSIVE = 1
    NORMAL = 2
    CAUTIOUS = 3
    POLITE = 4
    SUBMISSIVE = 5

class Vehicle:
    def __init__(self, id, position, velocity, lane, desired_velocity, driver_type=DriverType.NORMAL, 
                 length=5.0, width=2.0, vis_height=0.5, vis_width=6, color=None):
        self.id = id
        self.position = position  # longitudinal position (m)
        self.velocity = velocity  # current velocity (m/s)
        self.lane = lane  # current lane
        self.desired_velocity = desired_velocity  # desired velocity (m/s)
        self.length = length  # actual vehicle length (m)
        self.width = width  # actual vehicle width (m)
        self.vis_height = vis_height  # visualization height (relative to lane)
        self.vis_width = vis_width  # visualization width (relative to road)
        self.acceleration = 0.0  # current acceleration (m/s^2)
        self.driver_type = driver_type
        
        # Set parameters based on driver type
        self.set_driver_parameters()
        
        # Assign color based on driver type if none provided
        if color is None:
            self.color = self.get_driver_color()
        else:
            self.color = color
            
    def set_driver_parameters(self):
        """Set IDM and MOBIL parameters based on driver type."""
        if self.driver_type == DriverType.AGGRESSIVE:
            # Aggressive drivers: short following distance, not polite
            self.time_headway = 0.8  # very short desired time headway (s)
            self.min_gap = 1.5  # minimum gap (m)
            self.max_acceleration = 2.0  # higher acceleration
            self.comfortable_deceleration = 3.0  # more aggressive braking
            self.delta = 4.0  # acceleration exponent
            
            # MOBIL parameters
            self.politeness = 0.1  # very impolite
            self.changing_threshold = 0.0  # will change lanes for any advantage
            self.safe_deceleration = 5.0  # higher tolerance for unsafe situations
            self.right_bias = 0.1  # less bias to right lane
            
        elif self.driver_type == DriverType.NORMAL:
            # Normal drivers: average parameters
            self.time_headway = 1.5  # desired time headway (s)
            self.min_gap = 2.0  # minimum gap (m)
            self.max_acceleration = 1.5  # maximum acceleration (m/s^2)
            self.comfortable_deceleration = 2.0  # comfortable deceleration (m/s^2)
            self.delta = 4.0  # acceleration exponent
            
            # MOBIL parameters
            self.politeness = 0.3  # somewhat polite
            self.changing_threshold = 0.1  # acceleration gain threshold for lane change
            self.safe_deceleration = 4.0  # maximum safe deceleration
            self.right_bias = 0.3  # bias towards right lane
            
        elif self.driver_type == DriverType.CAUTIOUS:
            # Cautious drivers: long following distance, normal politeness
            self.time_headway = 2.2  # longer desired time headway (s)
            self.min_gap = 3.0  # larger minimum gap (m)
            self.max_acceleration = 1.2  # gentler acceleration
            self.comfortable_deceleration = 1.5  # more comfortable braking
            self.delta = 4.0  # acceleration exponent
            
            # MOBIL parameters
            self.politeness = 0.3  # normal politeness
            self.changing_threshold = 0.2  # higher threshold for lane changes
            self.safe_deceleration = 3.0  # lower tolerance for unsafe situations
            self.right_bias = 0.4  # stronger bias to right lane
            
        elif self.driver_type == DriverType.POLITE:
            # Polite drivers: normal following distance, very polite
            self.time_headway = 1.5  # desired time headway (s)
            self.min_gap = 2.0  # minimum gap (m)
            self.max_acceleration = 1.5  # maximum acceleration (m/s^2)
            self.comfortable_deceleration = 2.0  # comfortable deceleration (m/s^2)
            self.delta = 4.0  # acceleration exponent
            
            # MOBIL parameters
            self.politeness = 0.7  # very polite
            self.changing_threshold = 0.2  # higher threshold for lane changes
            self.safe_deceleration = 4.0  # maximum safe deceleration
            self.right_bias = 0.4  # stronger bias to right lane
            
        elif self.driver_type == DriverType.SUBMISSIVE:
            # Submissive drivers: long following distance, very polite
            self.time_headway = 2.5  # very long desired time headway (s)
            self.min_gap = 3.5  # large minimum gap (m)
            self.max_acceleration = 1.0  # gentle acceleration
            self.comfortable_deceleration = 1.5  # very comfortable braking
            self.delta = 4.0  # acceleration exponent
            
            # MOBIL parameters
            self.politeness = 0.8  # extremely polite
            self.changing_threshold = 0.3  # very high threshold for lane changes
            self.safe_deceleration = 2.5  # very low tolerance for unsafe situations
            self.right_bias = 0.5  # very strong bias to right lane
    
    def get_driver_color(self):
        """Return color based on driver type."""
        if self.driver_type == DriverType.AGGRESSIVE:
            return (0.8, 0.2, 0.2)  # red
        elif self.driver_type == DriverType.NORMAL:
            return (0.2, 0.6, 0.2)  # green
        elif self.driver_type == DriverType.CAUTIOUS:
            return (0.2, 0.2, 0.8)  # blue
        elif self.driver_type == DriverType.POLITE:
            return (0.8, 0.8, 0.2)  # yellow
        elif self.driver_type == DriverType.SUBMISSIVE:
            return (0.6, 0.2, 0.8)  # purple
    
    def idm_acceleration(self, lead_vehicle=None, road_length=1000):
        """Calculate acceleration based on IDM model."""
        # Free road acceleration
        a_free = self.max_acceleration * (1 - (self.velocity / self.desired_velocity) ** self.delta)
        
        if lead_vehicle is None:
            return a_free
        
        # Calculate gap and velocity difference
        gap = lead_vehicle.position - self.position - lead_vehicle.length
        
        # Handle circular boundary
        if gap < 0:
            gap += road_length
            
        delta_v = self.velocity - lead_vehicle.velocity
        
        # Calculate desired gap
        s_star = self.min_gap + max(0, self.velocity * self.time_headway + 
                                   (self.velocity * delta_v) / 
                                   (2 * np.sqrt(self.max_acceleration * self.comfortable_deceleration)))
        
        # Calculate interaction deceleration
        a_int = -self.max_acceleration * (s_star / max(gap, 0.1)) ** 2
        
        # Combine free and interaction accelerations
        return a_free + a_int
    
    def mobil_decide_lane_change(self, vehicles, lanes_count, road_length):
        """Decide if the vehicle should change lane based on MOBIL model."""
        # Get current acceleration
        current_acc = self.acceleration
        
        # Find relevant vehicles (leading and following) in current lane
        lead_current, follow_current = self.find_neighbors(vehicles, self.lane, road_length)
        
        # Check both adjacent lanes
        possible_lanes = []
        if self.lane > 0:
            possible_lanes.append(self.lane - 1)  # left lane
        if self.lane < lanes_count - 1:
            possible_lanes.append(self.lane + 1)  # right lane
            
        best_lane = self.lane
        max_advantage = 0
        
        for target_lane in possible_lanes:
            # Find relevant vehicles in target lane
            lead_target, follow_target = self.find_neighbors(vehicles, target_lane, road_length)
            
            # Safety criterion
            if not self.is_lane_change_safe(lead_target, follow_target, road_length):
                continue
                
            # Calculate advantage of lane change
            advantage = self.calculate_lane_change_advantage(
                current_acc, lead_current, follow_current, 
                lead_target, follow_target, target_lane, road_length
            )
            
            # Add right-lane bias (prefer to be in right lane)
            if target_lane > self.lane:
                advantage += self.right_bias
                
            if advantage > max_advantage and advantage > self.changing_threshold:
                max_advantage = advantage
                best_lane = target_lane
                
        return best_lane
    
    def find_neighbors(self, vehicles, lane, road_length):
        """Find leading and following vehicles in the specified lane."""
        lead_vehicle = None
        min_lead_distance = float('inf')
        
        follow_vehicle = None
        min_follow_distance = float('inf')
        
        for vehicle in vehicles:
            if vehicle.id == self.id or vehicle.lane != lane:
                continue
                
            # Calculate distance (accounting for circular road)
            distance = vehicle.position - self.position
            
            # Adjust for circular boundary
            if distance > road_length / 2:
                distance -= road_length
            elif distance < -road_length / 2:
                distance += road_length
                
            if distance > 0 and distance < min_lead_distance:
                min_lead_distance = distance
                lead_vehicle = vehicle
                
            if distance < 0 and abs(distance) < min_follow_distance:
                min_follow_distance = abs(distance)
                follow_vehicle = vehicle
                
        return lead_vehicle, follow_vehicle
    
    def is_lane_change_safe(self, lead_target, follow_target, road_length):
        """Check if lane change is safe."""
        # Check safety with respect to new leader
        if lead_target:
            gap = lead_target.position - self.position - lead_target.length
            
            # Adjust for circular boundary
            if gap < 0:
                gap += road_length
                
            if gap < self.min_gap:
                return False
                
        # Check safety with respect to new follower
        if follow_target:
            # Calculate acceleration of follower if we change lanes
            self_clone = Vehicle(
                id=-1,
                position=self.position,
                velocity=self.velocity,
                lane=follow_target.lane,
                desired_velocity=self.desired_velocity,
                length=self.length,
                driver_type=self.driver_type
            )
            
            new_follower_acc = follow_target.idm_acceleration(lead_vehicle=self_clone)
            
            if new_follower_acc < -self.safe_deceleration:
                return False
                
        return True
    
    def calculate_lane_change_advantage(self, current_acc, lead_current, follow_current, 
                                       lead_target, follow_target, target_lane, road_length):
        """Calculate the advantage of changing to the target lane."""
        # Calculate acceleration in new lane
        self_clone = Vehicle(
            id=-1,
            position=self.position,
            velocity=self.velocity,
            lane=target_lane,
            desired_velocity=self.desired_velocity,
            length=self.length,
            driver_type=self.driver_type
        )
        
        new_acc = self_clone.idm_acceleration(lead_vehicle=lead_target)
        acc_gain = new_acc - current_acc
        
        # Calculate disadvantage to the new follower
        disadvantage_follower = 0
        if follow_target:
            # Follower acceleration before lane change
            old_follower_acc = follow_target.idm_acceleration(lead_vehicle=lead_target)
            
            # Follower acceleration after lane change
            new_follower_acc = follow_target.idm_acceleration(lead_vehicle=self_clone)
            
            disadvantage_follower = old_follower_acc - new_follower_acc
            
        # Calculate disadvantage to the old follower
        disadvantage_old_follower = 0
        if follow_current:
            # Old follower acceleration before lane change
            old_acc = follow_current.idm_acceleration(lead_vehicle=self)
            
            # Old follower acceleration after lane change (no lead)
            new_acc = follow_current.idm_acceleration(lead_vehicle=lead_current)
            
            disadvantage_old_follower = old_acc - new_acc
            if disadvantage_old_follower < 0:  # this is actually an advantage
                disadvantage_old_follower = 0
                
        # Calculate total advantage using MOBIL equation
        total_advantage = acc_gain - self.politeness * (disadvantage_follower + disadvantage_old_follower)
        
        return total_advantage
        
    def update(self, dt, vehicles, lanes_count, road_length, change_lanes=True):
        """Update vehicle position, velocity, and lane."""
        # Find lead vehicle in current lane
        lead_vehicle, _ = self.find_neighbors(vehicles, self.lane, road_length)
        
        # Calculate acceleration using IDM
        self.acceleration = self.idm_acceleration(lead_vehicle, road_length)
        
        # Update velocity
        self.velocity += self.acceleration * dt
        self.velocity = max(0, min(self.velocity, 2 * self.desired_velocity))  # limit velocity
        
        # Update position
        self.position += self.velocity * dt
        
        # Handle circular boundary
        if self.position > road_length:
            self.position -= road_length
            
        # Consider lane change (if allowed)
        if change_lanes and random.random() < 0.1:  # Only consider lane changes occasionally
            new_lane = self.mobil_decide_lane_change(vehicles, lanes_count, road_length)
            self.lane = new_lane