#!/bin/bash

conda activate executorch
cd ${TORCH_PATH}/executorch

# Remove existing cmake-out directory
rm -rf cmake-out

# Create cmake-out directory
mkdir cmake-out

# Change to cmake-out directory
cd cmake-out

# Run CMake configuration
cmake .. -DCMAKE_BUILD_TYPE=Release -DEXECUTORCH_BUILD_SDK=ON -DBUCK2=${TORCH_PATH}/buck2e

# Check if CMake configuration was successful
if [ $? -eq 0 ]; then
    # Build the project
    cmake --build . -j12
else
    echo "CMake configuration failed. Exiting..."
fi
