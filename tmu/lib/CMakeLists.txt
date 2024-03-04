cmake_minimum_required(VERSION 3.15...3.27)
project(tumlibpp)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF) # Ensure -std=c++20 rather than -std=gnu++20

# Optimize for Release build
set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE) # Focus on Release build

# Detect the installed nanobind package and import it into CMake
execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# High performance and optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -ffast-math -march=native -DNDEBUG -flto -funroll-loops")

add_library(
        tmulib
        STATIC
        src/Attention.c
        src/ClauseBank.c
        src/ClauseBankSparse.c
        src/WeightBank.c
        src/Tools.c
        src/random/pcg32_fast.c
        src/random/xorshift128.c
        cpp/include/utils/sparse_clause_container.h
)
target_include_directories(tmulib PUBLIC include)
target_compile_options(tmulib PRIVATE
        $<$<CONFIG:Release>:-Ofast -ffast-math -march=native -DNDEBUG -flto -funroll-loops>
        $<$<CONFIG:Debug>:-O0 -g3 -DDEBUG -fsanitize=address>
)
target_link_libraries(
        tmulib
        PUBLIC
        $<$<CONFIG:Debug>: -fsanitize=address>
)


add_library(
        tmulibpp
        SHARED
        cpp/src/memory.cpp
)
target_include_directories(
        tmulibpp
        PUBLIC
        cpp/include
)
target_link_libraries(
        tmulibpp
        PUBLIC
        tmulib
)
target_compile_options(tmulibpp
        PRIVATE
        $<$<CONFIG:Release>:-Ofast -ffast-math -march=native -DNDEBUG -flto -funroll-loops >
        $<$<CONFIG:Debug>:-O0 -g3 -DDEBUG -fsanitize=address>
)

add_executable(
        tmulib_exe
        cpp/main.cpp
)
target_link_libraries(tmulib_exe PRIVATE tmulibpp)
target_compile_options(tmulib_exe PRIVATE
        $<$<CONFIG:Release>:-Ofast -ffast-math -march=native -DNDEBUG -flto>
        $<$<CONFIG:Debug>:-O0 -g3 -DDEBUG -fsanitize=address>
)

nanobind_add_module(
        tmulibpy
        NOMINSIZE
        cpp/nb/tmulibcpp.cpp
)
target_link_libraries(tmulibpy PRIVATE tmulibpp)
target_compile_options(tmulibpy PRIVATE
        $<$<CONFIG:Release>:-Os -ffast-math -march=native -DNDEBUG -flto>)


# Ensure CMake build type is set to Release for optimizations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
