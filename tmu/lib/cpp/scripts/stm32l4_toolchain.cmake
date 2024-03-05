# Set the system name to Generic, indicating a bare-metal target
set(CMAKE_SYSTEM_NAME Generic)


# Specify the cross-compiler locations
set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)

# Specify the processor and architecture flags
# Adjust these flags according to your specific STM32L4 model
set(CPU_FLAGS "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard")
set(COMMON_FLAGS "${CPU_FLAGS} -ffunction-sections -fdata-sections -MD -Wall -ffreestanding -fpermissive")
set(COMMON_FLAGS "${COMMON_FLAGS} -specs=nosys.specs")


set(CMAKE_C_FLAGS_INIT "${COMMON_FLAGS} -std=gnu11")
set(CMAKE_CXX_FLAGS_INIT "${COMMON_FLAGS} -std=gnu++14")
set(CMAKE_ASM_FLAGS_INIT "${CPU_FLAGS}")

# Set linker flags
# You will need to specify the correct linker script for your MCU model
# Get path of current fil
get_filename_component(CURRENT_DIR ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# Set the linker script
set(CMAKE_EXE_LINKER_FLAGS_INIT "-T${CURRENT_DIR}/STM32L475VGTX_FLASH.ld -Wl,--gc-sections")

# Optionally, you can set the build type to Release to optimize the binary size and performance
set(CMAKE_BUILD_TYPE_INIT Release)

# You may need to specify the path to the CMSIS and STM32L4 HAL library include directories
#include_directories(
#        /path/to/CMSIS/Include
#        /path/to/STM32L4xx_HAL_Driver/Inc
#)

# And, specify library search paths if you are linking against libraries
#link_directories(
#        /path/to/your/libraries
#)
