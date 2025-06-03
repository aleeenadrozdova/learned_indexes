cmake_minimum_required(VERSION 3.10)
project(IndexStructureBenchmark)

set(CMAKE_CXX_STANDARD 17)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Create directories for results
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/results")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/results/performance")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/results/memory")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/data")

# Add Python scripts
configure_file(src/python/train_rmi.py "${CMAKE_BINARY_DIR}/python/train_rmi.py" COPYONLY)
configure_file(src/python/train_fiting.py "${CMAKE_BINARY_DIR}/python/train_fiting.py" COPYONLY)
configure_file(src/python/train_radix.py "${CMAKE_BINARY_DIR}/python/train_radix.py" COPYONLY)

# Add executable
add_executable(benchmark src/cpp/main.cpp)

# Install Python dependencies
add_custom_target(
    python_deps ALL
    COMMAND ${Python3_EXECUTABLE} -m pip install numpy scikit-learn
    COMMENT "Installing Python dependencies"
)

# Add benchmark targets
add_custom_target(
    run_benchmarks
    COMMAND ${CMAKE_BINARY_DIR}/benchmark
    DEPENDS benchmark python_deps
    COMMENT "Running all benchmarks"
)


# Add a target for each distribution and size
foreach(DIST uniform normal lognormal zipf)
    foreach(SIZE 100000 1000000 10000000)
        add_custom_target(
            benchmark_${DIST}_${SIZE}
            COMMAND ${CMAKE_BINARY_DIR}/benchmark ${DIST} ${SIZE}
            DEPENDS benchmark python_deps
            COMMENT "Running benchmark for ${DIST} distribution with ${SIZE} elements"
        )
    endforeach()
endforeach()