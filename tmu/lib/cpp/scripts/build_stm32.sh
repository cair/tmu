#!/bin/bash

# Get the current directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Pull the Docker image with the STM32 toolchain
#docker pull lpodkalicki/stm32-toolchain

# Move to the project root directory; adjust this as necessary to point to your project's root
cd "$DIR/../../"

# Create the build directory if it doesn't exist and move into it
mkdir -p build-stm32



# Run a Docker container with the project directory mounted into /workspace
# Adjust the working directory to /workspace/build-stm32 to match the new build directory
docker run -v "$(pwd)":/workspace -w /workspace -it lpodkalicki/stm32-toolchain /bin/sh -c "\
ls -l && \
cd build-stm32 && rm -rf CMakeFiles && rm -f CMakeCache.txt && cd .. && \
TOOLCHAIN_FILE=\$(realpath ./cpp/scripts/stm32l4_toolchain.cmake) && \
cmake -S . -B build-stm32 \
  -DBUILD_STM32=ON \
  -DBUILD_PYTHON=OFF \
  -DBUILD_EXECUTABLE=ON \
  -DMCU_FAMILY=STM32L4 \
  -DMCU_MODEL=STM32L475VG \
  -DCMAKE_TOOLCHAIN_FILE=\$TOOLCHAIN_FILE \
  && cmake --build build-stm32 \
"
