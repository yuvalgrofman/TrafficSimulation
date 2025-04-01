import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from simulation import Simulation

class Visualization:
    def __init__(self, simulation: Simulation, update_interval: int = 50, visible_length: float = 4000):
        self.simulation = simulation
        self.update_interval = update_interval
        self.visible_length = visible_length

        self.vehicle_markers = {"car": "s", "truck": "^", "bus": "o"}
        self.driver_colors = {"aggressive": "red", "normal": "blue", "cautious": "green"}
        
        self.speed_cmap = LinearSegmentedColormap.from_list(
            'speed_colors', ['green', 'yellow', 'red'])
        
        self.fig = plt.figure(figsize=(14, 8))
        gs = plt.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[2, 1])
        
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax.set_xlim(0, self.visible_length)
        self.ax.set_xlabel('Position (m)')
        self.ax.set_yticks([])
        self.ax.set_title('Traffic Simulation')
        
        self.ax_speed = self.fig.add_subplot(gs[1, 0])
        self.ax_speed.set_xlabel("Car ID")
        self.ax_speed.set_ylabel("Speed (km/h)")
        self.ax_speed.set_title("Car Speeds", pad=10)
        
        self.cax = inset_axes(self.ax_speed, width="5%", height="80%", loc='center right',
                              bbox_to_anchor=(0.15, 0, 1, 1), bbox_transform=self.ax_speed.transAxes)
        
        self.speed_sm = plt.cm.ScalarMappable(cmap=self.speed_cmap, norm=plt.Normalize(vmin=0, vmax=120))
        self.colorbar = self.fig.colorbar(self.speed_sm, cax=self.cax, orientation='vertical', label='Speed (km/h)')
        
        self.car_artists = []
        self.status_text = self.fig.text(0.02, 0.98, "", fontsize=10)
        
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', label='Car', markerfacecolor='black', markersize=10),
            plt.Line2D([0], [0], marker='^', color='w', label='Truck', markerfacecolor='black', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Bus', markerfacecolor='black', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Aggressive', markerfacecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='blue', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Cautious', markerfacecolor='green', markersize=10),
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', ncol=2)
        
        self.speed_bars = None
        self.anim = FuncAnimation(self.fig, self.update, interval=self.update_interval, frames=None, blit=False, cache_frame_data=False)
        self.road_drawn = False
        self.lanes = []

    def draw_road(self):
        self.ax.set_facecolor('gray')
        if not self.lanes:
            return
        
        lowest_lane = min(self.lanes)
        highest_lane = max(self.lanes)
        road_top = highest_lane * 3 + 1.5
        road_bottom = lowest_lane * 3 - 1.5
        
        self.ax.set_ylim(road_bottom, road_top)
        self.ax.plot([0, self.visible_length], [road_top, road_top], 'white', lw=3)
        self.ax.plot([0, self.visible_length], [road_bottom, road_bottom], 'white', lw=3)
        
        for lane in self.lanes:
            lane_y_center = lane * 3
            for x in range(0, int(self.visible_length), 100):
                self.ax.plot([x, x + 50], [lane_y_center + 0.9, lane_y_center + 0.9], 'w', lw=2)
                self.ax.plot([x, x + 50], [lane_y_center - 0.9, lane_y_center - 0.9], 'w', lw=2)

    def update(self, frame):
        if not self.road_drawn:
            initial_car_data = self.simulation.get_car_data()
            self.lanes = list(initial_car_data.keys())
            if self.lanes:
                self.draw_road()
                self.road_drawn = True
        
        for _ in range(5):
            self.simulation.update()
        
        for artist in self.car_artists:
            artist.remove()
        self.car_artists = []
        
        car_data = self.simulation.get_car_data()
        cars_count = sum(len(cars) for cars in car_data.values())
        velocities = [car.velocity * 3.6 for cars in car_data.values() for car in cars]
        status = f"Time: {self.simulation.time:.1f}s | Cars: {cars_count} | "
        if velocities:
            status += f"Avg Speed: {np.mean(velocities):.1f} km/h"
        self.status_text.set_text(status)
        
        for lane, cars_in_lane in car_data.items():
            for car in cars_in_lane:
                if car.position < self.visible_length:
                    y_pos = lane * 3
                    marker = self.vehicle_markers.get(car.vehicle_type, "s")
                    color = self.driver_colors.get(car.driver_type, "black")
                    scatter = self.ax.scatter(car.position, y_pos, s=100, c=color, marker=marker, edgecolors='black')
                    self.car_artists.append(scatter)
                    text = self.ax.text(car.position, y_pos + 0.2, str(int(car.id)), fontsize=10, ha='center', color='black')
                    self.car_artists.append(text)
        
        self.ax_speed.clear()
        visible_cars = []
        for lane, cars_in_lane in car_data.items():
            visible_cars.extend([car for car in cars_in_lane if car.position < self.visible_length])
        
        if visible_cars:
            car_ids = [str(int(car.id)) for car in visible_cars]
            car_speeds = [car.velocity * 3.6 for car in visible_cars]
            max_speed = max(car_speeds) if car_speeds else 120
            norm_speeds = [speed / max_speed for speed in car_speeds]
            bar_colors = [self.speed_cmap(norm) for norm in norm_speeds]
            
            self.speed_bars = self.ax_speed.bar(car_ids, car_speeds, color=bar_colors, edgecolor='black')
            self.speed_sm.set_norm(plt.Normalize(vmin=0, vmax=max(120, max_speed)))
            self.colorbar.update_normal(self.speed_sm)
            
            for bar in self.speed_bars:
                height = bar.get_height()
                self.ax_speed.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        self.ax_speed.set_xlabel("Car ID")
        self.ax_speed.set_ylabel("Speed (km/h)")
        self.ax_speed.set_title("Car Speeds", pad=10)
    
    def show(self):
        plt.tight_layout()
        plt.show()