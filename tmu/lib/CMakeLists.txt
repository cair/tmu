cmake_minimum_required(VERSION 3.10)

project(TMU)

# Add the include directory to the search path for headers
include_directories(include)

# Create a list of source files
set(SOURCES
        src/Attention.c
        src/ClauseBank.c
        src/ClauseBankSparse.c
        src/ClauseWeightBank.c
        src/Tools.c
        src/WeightBank.c
        )

# Add a target for running unit tests
add_executable(${PROJECT_NAME}_test ${SOURCES} test/test_main.c test/test_weight_bank.cpp test/test_tools.cpp)

# Include Google Test as an external project
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.13.0
)
FetchContent_MakeAvailable(googletest)

# Link the test executable with the Google Test library
target_link_libraries(${PROJECT_NAME}_test gtest gtest_main)

# Add a custom target for running unit tests
add_custom_target(run_tests COMMAND ${PROJECT_NAME}_test)

# Set compiler flags
set(CMAKE_C_FLAGS "-Wall -Wextra -Wpedantic")

# Optionally set a build type (e.g. Debug or Release)
set(CMAKE_BUILD_TYPE Debug)
