
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
find_package(CUDAToolkit REQUIRED)

set(NVSHMEM_HOME "$ENV{NVSHMEM_HOME}")

set(GALATIC_HOME "./deps/CombBLAS/install/include/GALATIC/")
set(COMBBLAS_HOME $ENV{COMBBLAS_HOME})

# Include directories
target_include_directories(SpSYRK PUBLIC include)
target_include_directories(SpSYRK PUBLIC "${NVSHMEM_HOME}/include")
target_include_directories(SpSYRK PUBLIC "$ENV{MPICH_DIR}/include")
target_include_directories(SpSYRK PUBLIC "${GALATIC_HOME}/include")
target_include_directories(SpSYRK PUBLIC "./deps/json/include")
target_include_directories(SpSYRK PUBLIC "${COMBBLAS_HOME}/include")

# Link NVSHMEM 
target_link_libraries(SpSYRK PUBLIC "-L${NVSHMEM_HOME}/lib" -lnvshmem -lrt)
set_target_properties(SpSYRK PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Link CUDA libs
target_link_libraries(SpSYRK PUBLIC CUDA::cublas CUDA::cusparse "-L$ENV{MPICH_DIR}/gtl/lib" -lmpi_gtl_cuda -fopenmp -ltbb)

# Link MPI
target_link_libraries(SpSYRK PUBLIC MPI::MPI_CXX)

macro(add_test name)
    add_executable(${name} test/${name}.cu)

    target_include_directories(${name} PUBLIC include)
    target_include_directories(${name} PUBLIC "${NVSHMEM_HOME}/include")
    target_include_directories(${name} PUBLIC "$ENV{MPICH_DIR}/include")
    target_include_directories(${name} PUBLIC "${GALATIC_HOME}/include")
    target_include_directories(${name} PUBLIC "./deps/json/include")
    target_include_directories(${name} PUBLIC "${COMBBLAS_HOME}/include")

    # Link NVSHMEM 
    target_link_libraries(${name} PUBLIC "-L${NVSHMEM_HOME}/lib" -lnvshmem -lrt)
    set_target_properties(${name} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    # Link CUDA libs
    target_link_libraries(${name} PUBLIC CUDA::cublas CUDA::cusparse "-L$ENV{MPICH_DIR}/gtl/lib" -lmpi_gtl_cuda -fopenmp -ltbb)

    # Link MPI
    target_link_libraries(${name} PUBLIC MPI::MPI_CXX)

    # CUDA flags
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -ccbin CC>)
endmacro()

macro(add_experiment name)
    add_executable(${name} experiments/${name}.cu)

    target_include_directories(${name} PUBLIC include)
    target_include_directories(${name} PUBLIC "${NVSHMEM_HOME}/include")
    target_include_directories(${name} PUBLIC "$ENV{MPICH_DIR}/include")
    target_include_directories(${name} PUBLIC "${GALATIC_HOME}/include")
    target_include_directories(${name} PUBLIC "./deps/json/include")
    target_include_directories(${name} PUBLIC "${COMBBLAS_HOME}/include")

    # Link NVSHMEM 
    target_link_libraries(${name} PUBLIC "-L${NVSHMEM_HOME}/lib" -lnvshmem -lrt)
    set_target_properties(${name} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    # Link CUDA libs
    target_link_libraries(${name} PUBLIC CUDA::cublas CUDA::cusparse "-L$ENV{MPICH_DIR}/gtl/lib" -lmpi_gtl_cuda -fopenmp -ltbb)

    # Link MPI
    target_link_libraries(${name} PUBLIC MPI::MPI_CXX)

    # CUDA flags
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -ccbin CC -fopenmp>)
endmacro()

macro(add_app name)
    add_executable(${name} applications/${name}.cu)

    target_include_directories(${name} PUBLIC include)
    target_include_directories(${name} PUBLIC "${NVSHMEM_HOME}/include")
    target_include_directories(${name} PUBLIC "$ENV{MPICH_DIR}/include")
    target_include_directories(${name} PUBLIC "${GALATIC_HOME}/include")
    target_include_directories(${name} PUBLIC "./deps/json/include")
    target_include_directories(${name} PUBLIC "${COMBBLAS_HOME}/include")

    # Link NVSHMEM 
    target_link_libraries(${name} PUBLIC "-L${NVSHMEM_HOME}/lib" -lnvshmem -lrt)
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
    target_include_directories(${name} PUBLIC "$ENV{MPICH_DIR}/include")
    target_include_directories(${name} PUBLIC "${GALATIC_HOME}/include")
    target_include_directories(${name} PUBLIC "./deps/json/include")
    target_include_directories(${name} PUBLIC "${COMBBLAS_HOME}/include")

    # Link NVSHMEM 
    target_link_libraries(${name} PUBLIC "-L${NVSHMEM_HOME}/lib" -lnvshmem -lrt)
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
add_experiment(run_combblas)
target_link_libraries(run_combblas PUBLIC "-L${COMBBLAS_HOME}/lib" -lCombBLAS -lGraphGenlib  -lUsortlib)

# Tests
add_test(hello_world)
add_test(test_transpose)
add_test(test_local_mult)
add_test(test_spsyrk_bulksync_1d)
add_test(test_spsyrk_cyclic_2d)

# Applications
add_app(overlap_detection)

