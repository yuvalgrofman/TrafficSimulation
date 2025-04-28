#!/bin/bash

# Base parameters
ROAD_LENGTH=1000
LANES=1
NUM_SIMULATIONS=20
VEHICLE_COUNTS="5,10,15,20,25,30,35,40,45,50,55,60"
DISTRACTED_PERCENTAGE=0
SIM_TIME=150  # Using the default value from earlier example

# Create parent output directory if it doesn't exist
BASE_OUTPUT_DIR="simulation_results/1L/normal_aggressive_00D"
mkdir -p "$BASE_OUTPUT_DIR"

# Loop through aggressive driver percentages from 0 to 100 in increments of 10
for AGGRESSIVE_PCT in $(seq 0 10 100); do
    # Calculate normal driver percentage (inverse of aggressive)
    NORMAL_PCT=$((100 - AGGRESSIVE_PCT))
    
    echo "Running simulation with driver distribution: $NORMAL_PCT% normal, $AGGRESSIVE_PCT% aggressive"
    
    # Create the driver distribution string
    # Format: normal,aggressive,cautious,distracted,reckless
    # We're only varying normal and aggressive, keeping others at 0
    DRIVER_DIST="0.$(printf "%02d" $NORMAL_PCT),0.$(printf "%02d" $AGGRESSIVE_PCT),0,0,0"
    
    # Create output directory for this driver distribution
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/normal_${NORMAL_PCT}_aggressive_${AGGRESSIVE_PCT}_distracted_${DISTRACTED_PERCENTAGE}"
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
    
    echo "Completed simulation with driver distribution: $NORMAL_PCT% normal, $AGGRESSIVE_PCT% aggressive"
    echo "Results saved in: $OUTPUT_DIR"
    echo "---------------------------------------------------"
done

echo "All simulations completed successfully!"