import numpy as np
from simulation import Simulation
from visualization import Visualization
from cars import Car, CarFactory

def run_simulation(simulation_time: float = 3600, visualization: bool = True):
    # Artifical cars
    car1 = CarFactory.create_vehicle("car", "normal", 0, 40, 1)
    car2 = CarFactory.create_vehicle("car", "normal", 0, 120, 1)
    car3 = CarFactory.create_vehicle("car", "normal", 0, 120, 1)

    # Set lanes
    car1.lane = 1
    car2.lane = 2
    car3.lane = 1

    # Initialize simulation
    car_flow = [
        {"car": car1, "inflow_time": 2},
        {"car": car2, "inflow_time": 1},
        {"car": car3, "inflow_time": 10}
    ]

    # Create simulation with a smaller time step for better accuracy
    sim = Simulation(dt=0.01, lane_changing=True, artificial_cars=car_flow)
    
    if visualization:
        # Run with visualization
        vis = Visualization(sim, update_interval=20)  # Faster refresh rate
        vis.show()
    else:
        # Run simulation without visualization
        steps = int(simulation_time / sim.dt)
        for i in range(steps):
            sim.update()
            # Print progress every 10%
            if i % (steps // 10) == 0:
                print(f"Simulation progress: {i / steps * 100:.0f}%")
            
        # Final statistics
        velocities = sim.get_car_velocities()
        if velocities:
            print(f"Final number of cars: {len(velocities)}")
            print(f"Average speed: {np.mean(velocities):.2f} km/h")
            print(f"Min speed: {min(velocities):.2f} km/h")
            print(f"Max speed: {max(velocities):.2f} km/h")


if __name__ == "__main__":
    # Run simulation with visualization
    import sys
    
    # Check if any command line arguments are provided
    visualization = True
    if len(sys.argv) > 1 and sys.argv[1].lower() == "novis":
        visualization = False
    
    run_simulation(simulation_time=3600, visualization=visualization)