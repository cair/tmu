cmake_minimum_required(VERSION 3.10)

project(TMU LANGUAGES C)

set(CMAKE_C_STANDARD 99)

include_directories(include)

add_executable(TMU
        main.c
        src/Attention.c
        src/ClauseBank.c
        src/ClauseBankSparse.c
        src/Tools.c
        src/WeightBank.c
)

target_link_libraries(TMU m)
