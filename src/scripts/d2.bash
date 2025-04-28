#!/bin/bash

# Base parameters
ROAD_LENGTH=500
LANES=2
NUM_SIMULATIONS=30
VEHICLE_COUNTS="5,10,15,20,25,30,35,40,45,50,55,60,65,70"
DRIVER_DIST="0,1,0,0,0"
SIM_TIME=150  # Using the default value from the example

# Create parent output directory if it doesn't exist
BASE_OUTPUT_DIR="simulation_results/2L_precise/2L_normal_distracted"
mkdir -p "$BASE_OUTPUT_DIR"

# Loop through distracted percentages from 0 to 100 in increments of 10
for DISTRACTED_PERCENTAGE in $(seq 0 10 50); do
    echo "Running simulation with distracted percentage: $DISTRACTED_PERCENTAGE%"
    
    # Create output directory for this distracted percentage
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/distracted_${DISTRACTED_PERCENTAGE}_percent"
    mkdir -p "$OUTPUT_DIR"
    
    # Run the simulation
    python src/main.py \
        --mode multiple \
        --road-length $ROAD_LENGTH \
        --lanes $LANES \
        --distracted-percentage $DISTRACTED_PERCENTAGE \
        --sim-time $SIM_TIME \
        --num-simulations $NUM_SIMULATIONS \
        --vehicle-counts "$VEHICLE_COUNTS" \
        --driver-distribution "$DRIVER_DIST" \
        --output-dir "$OUTPUT_DIR"
    
    echo "Completed simulation with distracted percentage: $DISTRACTED_PERCENTAGE%"
    echo "Results saved in: $OUTPUT_DIR"
    echo "---------------------------------------------------"
done

echo "All simulations completed successfully!"