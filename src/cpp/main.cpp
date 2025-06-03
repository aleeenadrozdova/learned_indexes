#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <fstream>
#include <memory>
#include <algorithm>
#include <functional>

#include "benchmark.h"
#include "data_generator.h"

// Writing data to file for Python script processing
void write_data_to_file(const std::vector<uint64_t>& data, const std::string& file_path) {
    std::ofstream file(file_path);
    for (const auto& key : data) {
        file << key << std::endl;
    }
    file.close();
}

// Executing Python script from C++
bool execute_python_script(const std::string& script_path, const std::string& args) {
    std::string command = "python3 " + script_path + " " + args;
    std::cout << "Executing: " << command << std::endl;
    int result = system(command.c_str());
    return result == 0;
}

// Training RMI model
bool train_rmi_model(const std::vector<uint64_t>& data, const std::string& data_type, size_t data_size) {
    std::string data_file = "data/" + data_type + "_" + std::to_string(data_size) + ".txt";
    std::string model_file = "models/rmi_" + data_type + "_" + std::to_string(data_size) + ".json";
    
    write_data_to_file(data, data_file);
    return execute_python_script("python/train_rmi.py", data_file + " " + model_file + " --num_models 100");
}

// Training FITing Tree model
bool train_fiting_tree_model(const std::vector<uint64_t>& data, const std::string& data_type, size_t data_size) {
    std::string data_file = "data/" + data_type + "_" + std::to_string(data_size) + ".txt";
    std::string model_file = "models/fiting_" + data_type + "_" + std::to_string(data_size) + ".pkl";
    
    write_data_to_file(data, data_file);
    return execute_python_script("python/train_fiting.py", data_file + " " + model_file + " --error_bound 32");
}

// Training RadixSpline model
bool train_radix_spline_model(const std::vector<uint64_t>& data, const std::string& data_type, size_t data_size) {
    std::string data_file = "data/" + data_type + "_" + std::to_string(data_size) + ".txt";
    std::string model_file = "models/radix_" + data_type + "_" + std::to_string(data_size) + ".pkl";
    
    write_data_to_file(data, data_file);
    return execute_python_script("python/train_radix.py", data_file + " " + model_file + " --error_bound 32 --radix_bits 18");
}

void run_benchmarks(const std::string& data_type, size_t data_size) {
    // Creating necessary directories
    system("mkdir -p data");
    system("mkdir -p models");
    system("mkdir -p results/performance");
    
    // Data generation
    std::vector<uint64_t> keys = generate_data(data_type, data_size);
    
    // Loading data into B-Tree and B+-Tree
    std::cout << "Loading data into indexes..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    BTree<uint64_t> btree;
    for (const auto& key : keys) {
        btree.insert(key);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> btree_build_time = end_time - start_time;
    std::cout << "B-Tree built successfully" << std::endl;
    
    start_time = std::chrono::high_resolution_clock::now();
    BPlusTree<uint64_t> bplus_tree;
    for (const auto& key : keys) {
        bplus_tree.insert(key);
    }
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> bplus_tree_build_time = end_time - start_time;
    std::cout << "B+-Tree built successfully" << std::endl;
    
    // Sorting data for trained indexes
    std::sort(keys.begin(), keys.end());
    
    // Training and loading RMI model
    std::cout << "Training RMI model..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    RMI<uint64_t> rmi;
    bool rmi_trained = train_rmi_model(keys, data_type, data_size);
    
    if (rmi_trained) {
        // Loading trained model into RMI
        std::string model_file = "models/rmi_" + data_type + "_" + std::to_string(data_size) + ".json";
        rmi.load_model(model_file);
        std::cout << "RMI model trained and loaded successfully" << std::endl;
    } else {
        // If Python script failed, use built-in training
        std::cout << "Failed to train model via Python, using built-in training" << std::endl;
        // rmi.build(keys);
    }
    rmi.load_data(keys);
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> rmi_build_time = end_time - start_time;
    
    // Training and loading FITing Tree model
    std::cout << "Training FITing Tree model..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    FitingTree<uint64_t> fiting_tree;
    fiting_tree.build(keys);
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> fiting_tree_build_time = end_time - start_time;
    
    // Training and loading RadixSpline model
    std::cout << "Training RadixSpline model..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    RadixSpline<uint64_t> radix_spline;
    radix_spline.build(keys);
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> radix_spline_build_time = end_time - start_time;
    
    // Writing build time to file
    std::ofstream build_time_file("results/performance/build_time_" + data_type + "_" + std::to_string(data_size) + ".csv");
    build_time_file << "Index,BuildTime(s)" << std::endl;
    append_to_results_file("B-Tree", data_type, data_size, "build_time", btree_build_time.count());
    append_to_results_file("B+-Tree", data_type, data_size, "build_time", bplus_tree_build_time.count());
    append_to_results_file("RMI", data_type, data_size, "build_time", rmi_build_time.count());
    append_to_results_file("FITing-Tree", data_type, data_size, "build_time", fiting_tree_build_time.count());
    append_to_results_file("RadixSpline", data_type, data_size, "build_time", radix_spline_build_time.count());
    build_time_file << "B-Tree," << btree_build_time.count() << std::endl;
    build_time_file << "B+-Tree," << bplus_tree_build_time.count() << std::endl;
    build_time_file << "RMI," << rmi_build_time.count() << std::endl;
    build_time_file << "FITing-Tree," << fiting_tree_build_time.count() << std::endl;
    build_time_file << "RadixSpline," << radix_spline_build_time.count() << std::endl;
    build_time_file.close();

     
    try {
            measure_memory_usage(btree, bplus_tree, rmi, fiting_tree, radix_spline, data_type, data_size);
        } catch (const std::exception& e) {
            // Exception handling
            std::cout << "Exception caught during memory usage measurement: " << e.what() << std::endl;
        } 
    
    // Further performance tests...
    // Testing search
    std::vector<uint64_t> search_keys = generate_search_keys(keys, 1000);
    run_search_benchmark(btree, bplus_tree, rmi, fiting_tree, radix_spline, search_keys, data_type, data_size);

    
    // Testing range search
    std::vector<std::pair<uint64_t, uint64_t>> range_queries = generate_range_queries(keys, 10);
    run_range_search_benchmark(btree, bplus_tree, rmi, fiting_tree, radix_spline, range_queries, data_type, data_size);

    
    // // Testing insertion and deletion
    // std::vector<uint64_t> new_keys = generate_data(data_type, 1000);
    // run_insert_benchmark(btree, bplus_tree, rmi, fiting_tree, radix_spline, new_keys, data_type, data_size);
    // run_delete_benchmark(btree, bplus_tree, rmi, fiting_tree, radix_spline, search_keys, data_type, data_size);

    // // Measuring memory usage

}

int main(int argc, char* argv[]) {
    std::vector<std::string> data_types = {"uniform", "normal", "zipf", "lognormal"};
    std::vector<size_t> data_sizes = {10000, 100000, 1000000, 10000000};
    
    for (const auto& data_size : data_sizes) {
        for (const auto& data_type : data_types) {
            std::cout << "Running benchmarks for " << data_type << " distribution with " << data_size << " elements..." << std::endl;
            try {
                run_benchmarks(data_type, data_size);
            } catch (const std::exception& e) {
                // Handling standard C++ exceptions
                std::cerr << "Error during benchmark execution: " << e.what() << std::endl;
            } catch (...) {
                // Handling any other exceptions
                std::cerr << "Unknown error during benchmark execution" << std::endl;
            }
        }
    }
    return 0;
}
