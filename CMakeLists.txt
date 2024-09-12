
cmake_minimum_required(VERSION 3.3)
project(SpSYRK VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -rdc=true -ccbin CC") 
set(CMAKE_CUDA_ARCHITECTURES "80")

#cmake_policy(SET CMP0074 OLD)


# Main library
add_library(SpSYRK src/main.cu)


# Compiler options
target_compile_options(SpSYRK PRIVATE -Wno-deprecated-declarations)

# Find CUDA and NVSHMEM 
find_package(CUDA)

set(NVSHMEM_HOME "/global/common/software/nersc9/nvshmem/2.11.0")
set(OPENSHMEM_HOME "/opt/cray/pe/sma/11.7.0/ofi/sma")

set(BCL_HOME "~/bcl")
set(GALATIC_HOME "~/GALATIC/")

# Include directories
target_include_directories(SpSYRK PUBLIC include)
target_include_directories(SpSYRK PUBLIC "${NVSHMEM_HOME}/include")
target_include_directories(SpSYRK PUBLIC "${OPENSHMEM_HOME}/include")
target_include_directories(SpSYRK PUBLIC "$ENV{MPICH_DIR}/include")
target_include_directories(SpSYRK PUBLIC "${GALATIC_HOME}/include")
target_include_directories(SpSYRK PUBLIC "${BCL_HOME}")

# Link NVSHMEM 
target_link_libraries(SpSYRK PUBLIC "-L${NVSHMEM_HOME}/lib" -lnvshmem -lrt)
target_link_libraries(SpSYRK PUBLIC "-L${OPENSHMEM_HOME}/lib64" -lsma)
set_target_properties(SpSYRK PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Link CUDA libs
target_link_libraries(SpSYRK PUBLIC "-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/lib64/" -lcublas -lcusparse -lcuda -lcudart "-L$ENV{MPICH_DIR}/gtl/lib" -lmpi_gtl_cuda)

# Tests
add_executable(hello_world test/hello_world.cu)

target_include_directories(hello_world PUBLIC include)

# Link NVSHMEM 
target_link_libraries(hello_world PUBLIC "-L${NVSHMEM_HOME}/lib" -lnvshmem -lrt)
target_link_libraries(hello_world PUBLIC "-L${OPENSHMEM_HOME}/lib64" -lsma)
set_target_properties(hello_world PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Link CUDA libs
target_link_libraries(hello_world PUBLIC "-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/lib64/" -lcublas -lcusparse -lcuda -lcudart )

