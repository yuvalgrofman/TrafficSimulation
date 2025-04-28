#!/bin/bash

# Base parameters
ROAD_LENGTH=1000
LANES=1
NUM_SIMULATIONS=20
VEHICLE_COUNTS="5,10,15,20,25,30,35,40,45,50,55,60"
DISTRACTED_PERCENTAGE=50
SIM_TIME=150  # Using the default value from earlier example

BASE_OUTPUT_DIR="simulation_results/test"
mkdir -p "$BASE_OUTPUT_DIR"

for AGGRESSIVE_PCT in $(seq 0 10 100); do
    NORMAL_PCT=$((100 - AGGRESSIVE_PCT))
    
    DRIVER_DIST="0.$(printf "%02d" $NORMAL_PCT),0.$(printf "%02d" $AGGRESSIVE_PCT),0,0,0"
    
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