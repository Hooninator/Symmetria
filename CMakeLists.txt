
cmake_minimum_required(VERSION 3.3)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -rdc=true -ccbin ${CC} --expt-relaxed-constexpr ") 
set(CMAKE_CUDA_ARCHITECTURES "80")
set(CMAKE_CXX_COMPILER "CC")
set(CMAKE_CUDA_HOST_COMPILER "CC")

#cmake_policy(SET CMP0074 OLD)
project(SpSYRK VERSION 1.0.0 LANGUAGES CXX CUDA)


# Main library
add_library(SpSYRK src/main.cu)


# Compiler options
target_compile_options(SpSYRK PRIVATE -Wno-deprecated-declarations -O3 -fopenmp)

# Find MPI
find_package(MPI REQUIRED)

# Find CUDA and NVSHMEM 
find_package(CUDA)

set(NVSHMEM_HOME "$ENV{NVSHMEM_HOME}")
set(OPENSHMEM_HOME "/opt/cray/pe/sma/11.7.0/ofi/sma")

set(BCL_HOME "~/bcl")
set(GALATIC_HOME "~/GALATIC/")
set(COMBBLAS_HOME "./deps/CombBLAS/install/")

# Include directories
target_include_directories(SpSYRK PUBLIC include)
target_include_directories(SpSYRK PUBLIC "${NVSHMEM_HOME}/include")
target_include_directories(SpSYRK PUBLIC "${OPENSHMEM_HOME}/include")
target_include_directories(SpSYRK PUBLIC "$ENV{MPICH_DIR}/include")
target_include_directories(SpSYRK PUBLIC "${GALATIC_HOME}/include")
target_include_directories(SpSYRK PUBLIC "${BCL_HOME}")
target_include_directories(SpSYRK PUBLIC "./deps/json/include")
target_include_directories(SpSYRK PUBLIC "${COMBBLAS_HOME}/include")

# Link NVSHMEM 
target_link_libraries(SpSYRK PUBLIC "-L${NVSHMEM_HOME}/lib" -lnvshmem -lrt)
target_link_libraries(SpSYRK PUBLIC "-L${OPENSHMEM_HOME}/lib64" -lsma)
set_target_properties(SpSYRK PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Link CUDA libs
target_link_libraries(SpSYRK PUBLIC "-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/lib64/" -lcublas -lcusparse -lcuda -lcudart "-L$ENV{MPICH_DIR}/gtl/lib" -lmpi_gtl_cuda -ltbb -fopenmp)

# Link MPI
target_link_libraries(SpSYRK PUBLIC MPI::MPI_CXX)

macro(add_test name)
    add_executable(${name} test/${name}.cu)

    target_include_directories(${name} PUBLIC include)
    target_include_directories(${name} PUBLIC "${NVSHMEM_HOME}/include")
    target_include_directories(${name} PUBLIC "${OPENSHMEM_HOME}/include")
    target_include_directories(${name} PUBLIC "$ENV{MPICH_DIR}/include")
    target_include_directories(${name} PUBLIC "${GALATIC_HOME}/include")
    target_include_directories(${name} PUBLIC "${BCL_HOME}")
    target_include_directories(${name} PUBLIC "./deps/json/include")
    target_include_directories(${name} PUBLIC "${COMBBLAS_HOME}/include")

    # Link NVSHMEM 
    target_link_libraries(${name} PUBLIC "-L${NVSHMEM_HOME}/lib" -lnvshmem -lrt)
    target_link_libraries(${name} PUBLIC "-L${OPENSHMEM_HOME}/lib64" -lsma)
    set_target_properties(${name} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    # Link CUDA libs
    target_link_libraries(${name} PUBLIC "-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/lib64/" -lcublas -lcusparse -lcuda -lcudart "-L$ENV{MPICH_DIR}/gtl/lib" -lmpi_gtl_cuda -ltbb )

    # Link MPI
    target_link_libraries(${name} PUBLIC MPI::MPI_CXX)

    # CUDA flags
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -ccbin CC>)
endmacro()

macro(add_experiment name)
    add_executable(${name} experiments/${name}.cu)

    target_include_directories(${name} PUBLIC include)
    target_include_directories(${name} PUBLIC "${NVSHMEM_HOME}/include")
    target_include_directories(${name} PUBLIC "${OPENSHMEM_HOME}/include")
    target_include_directories(${name} PUBLIC "$ENV{MPICH_DIR}/include")
    target_include_directories(${name} PUBLIC "${GALATIC_HOME}/include")
    target_include_directories(${name} PUBLIC "${BCL_HOME}")
    target_include_directories(${name} PUBLIC "./deps/json/include")
    target_include_directories(${name} PUBLIC "${COMBBLAS_HOME}/include")

    # Link NVSHMEM 
    target_link_libraries(${name} PUBLIC "-L${NVSHMEM_HOME}/lib" -lnvshmem -lrt)
    target_link_libraries(${name} PUBLIC "-L${OPENSHMEM_HOME}/lib64" -lsma)
    set_target_properties(${name} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    # Link CUDA libs
    target_link_libraries(${name} PUBLIC "-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/lib64/" -lcublas -lcusparse -lcuda -lcudart "-L$ENV{MPICH_DIR}/gtl/lib" -lmpi_gtl_cuda -ltbb -fopenmp)

    # Link MPI
    target_link_libraries(${name} PUBLIC MPI::MPI_CXX)

    # CUDA flags
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -ccbin CC -fopenmp>)
endmacro()

# Experiments
add_experiment(run_spsyrk)


# Tests
add_test(hello_world)
add_test(test_transpose)
add_test(test_local_mult)
add_test(test_spsyrk_bulksync_1d)
add_test(test_spsyrk_cyclic_2d)

