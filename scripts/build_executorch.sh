#!/bin/bash
# https://pytorch.org/executorch/stable/runtime-build-and-cross-compilation.html

cd ${TORCH_PATH}/executorch

# Remove existing cmake-out directory
sudo rm -rf cmake-out

# Create cmake-out directory
mkdir cmake-out
cd cmake-out

# Run CMake configuration
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_SDK=ON \
    -DBUCK2=${TORCH_PATH}/buck2e \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DPYTHON_EXECUTABLE=python3 \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON

# Check if CMake configuration was successful
if [ $? -eq 0 ]; then
    # Build the project
    sudo cmake --build . -j13 --target install --config Release
else
    echo "CMake configuration failed. Exiting..."
fi
