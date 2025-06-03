#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <functional>

#include "btree.h"
#include "bplus_tree.h"
#include "rmi.h"
#include "fiting_tree.h"
#include "radix_spline.h"

// Write result into a file
template <typename DurationType>
void append_to_results_file(
    const std::string& index_name,
    const std::string& data_type,
    size_t data_size,
    const std::string& operation,
    DurationType duration_ns,
    const std::string& filename = "/Users/adrozdovam/Documents/coursework/results/benchmark_results.csv"
) {
    static bool file_exists = true;
    std::ofstream result_file;
    
    if (!file_exists) {
        // Create file and write the header
        std::filesystem::create_directories(
        std::filesystem::path(filename).parent_path());
        result_file.open(filename);
        result_file << "Index,DistributionType,DataSize,Operation,Time(ns)" << std::endl;
        file_exists = true;
    } else {
        result_file.open(filename, std::ios::app);
    }
    
    result_file << index_name << ","
                << data_type << ","
                << data_size << ","
                << operation << ","
                << duration_ns << std::endl;
    
    result_file.close();
}

// Benchmark the search operation
template <typename T>
void run_search_benchmark(
    const BTree<T>& btree,
    const BPlusTree<T>& bplus_tree,
    const RMI<T>& rmi,
    const FitingTree<T>& fiting_tree,
    const RadixSpline<T>& radix_spline,
    const std::vector<T>& search_keys,
    const std::string& data_type,
    size_t data_size
) {
    std::cout << "Running search benchmark for " << data_type << " with " << data_size << " elements..." << std::endl;
    
    // Benchmark B-Tree
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t found_count = 0;
    for (const auto& key : search_keys) {
        if (btree.search(key)) {
            found_count++;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / search_keys.size();
    append_to_results_file("B-Tree", data_type, data_size, "search", duration);
    std::cout << "B-Tree search: " << duration << " ns/op, found " << found_count << " keys" << std::endl;
    
    // Benchmark B+-Tree
    start_time = std::chrono::high_resolution_clock::now();
    found_count = 0;
    for (const auto& key : search_keys) {
        if (bplus_tree.search(key)) {
            found_count++;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / search_keys.size();
    append_to_results_file("B+-Tree", data_type, data_size, "search", duration);
    std::cout << "B+-Tree search: " << duration << " ns/op, found " << found_count << " keys" << std::endl;
    
    // Benchmark RMI
    start_time = std::chrono::high_resolution_clock::now();
    found_count = 0;
    for (const auto& key : search_keys) {
        try {
            if (rmi.lookup(key) >= 0) {
                found_count++;
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception in RMI lookup: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception in RMI lookup" << std::endl;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / search_keys.size();
    append_to_results_file("RMI", data_type, data_size, "search", duration);
    std::cout << "RMI search: " << duration << " ns/op, found " << found_count << " keys" << std::endl;
    
    // Benchmark FITing-Tree
    start_time = std::chrono::high_resolution_clock::now();
    found_count = 0;
    for (const auto& key : search_keys) {
        try {
            if (fiting_tree.lookup(key) >= 0) {
                found_count++;
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception in FITing-Tree lookup: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception in FITing-Tree lookup" << std::endl;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / search_keys.size();
    append_to_results_file("FITing-Tree", data_type, data_size, "search", duration);
    std::cout << "FITing-Tree search: " << duration << " ns/op, found " << found_count << " keys" << std::endl;
    
    // Benchmark RadixSpline
    start_time = std::chrono::high_resolution_clock::now();
    found_count = 0;
    for (const auto& key : search_keys) {
        try {
            if (radix_spline.lookup(key) >= 0) {
                found_count++;
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception in RadixSpline lookup: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception in RadixSpline lookup" << std::endl;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / search_keys.size();
    append_to_results_file("RadixSpline", data_type, data_size, "search", duration);
    std::cout << "RadixSpline search: " << duration << " ns/op, found " << found_count << " keys" << std::endl;
}

// Benchmark the range search operation
template <typename T>
void run_range_search_benchmark(
    const BTree<T>& btree,
    const BPlusTree<T>& bplus_tree,
    const RMI<T>& rmi,
    const FitingTree<T>& fiting_tree,
    const RadixSpline<T>& radix_spline,
    const std::vector<std::pair<T, T>>& range_queries,
    const std::string& data_type,
    size_t data_size
) {
    std::cout << "Running range search benchmark for " << data_type << " with " << data_size << " elements..." << std::endl;
    
    // Benchmark B-Tree
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t total_results = 0;
    for (const auto& query : range_queries) {
        auto results = btree.range_search(query.first, query.second);
        total_results += results.size();
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / range_queries.size();
    append_to_results_file("B-Tree", data_type, data_size, "range_search", duration);
    std::cout << "B-Tree range search: " << duration << " ns/op, found " << total_results << " keys" << std::endl;
    
    // Benchmark B+-Tree
    start_time = std::chrono::high_resolution_clock::now();
    total_results = 0;
    for (const auto& query : range_queries) {
        auto results = bplus_tree.range_search(query.first, query.second);
        total_results += results.size();
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / range_queries.size();
    append_to_results_file("B+-Tree", data_type, data_size, "range_search", duration);
    std::cout << "B+-Tree range search: " << duration << " ns/op, found " << total_results << " keys" << std::endl;
    
    // Benchmark RMI
    start_time = std::chrono::high_resolution_clock::now();
    total_results = 0;
    for (const auto& query : range_queries) {
        try {
            auto results = rmi.range_query(query.first, query.second);
            total_results += results.size();
        } catch (const std::exception& e) {
            std::cerr << "Exception in RMI range_query: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception in RMI range_query" << std::endl;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / range_queries.size();
    append_to_results_file("RMI", data_type, data_size, "range_search", duration);
    std::cout << "RMI range search: " << duration << " ns/op, found " << total_results << " keys" << std::endl;

   // Benchmark FITing-Tree
    start_time = std::chrono::high_resolution_clock::now();
    total_results = 0;
    for (const auto& query : range_queries) {
        try {
            auto results = fiting_tree.range_query(query.first, query.second);
            total_results += results.size();
        } catch (const std::exception& e) {
            std::cerr << "Exception in FITing-Tree range_query: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception in FITing-Tree range_query" << std::endl;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / range_queries.size();
    append_to_results_file("FITing-Tree", data_type, data_size, "range_search", duration);
    std::cout << "FITing-Tree range search: " << duration << " ns/op, found " << total_results << " keys" << std::endl;
    
    // Benchmark RadixSpline
    start_time = std::chrono::high_resolution_clock::now();
    total_results = 0;
    for (const auto& query : range_queries) {
        try {
            auto results = radix_spline.range_query(query.first, query.second);
            total_results += results.size();
        } catch (const std::exception& e) {
            std::cerr << "Exception in RadixSpline range_query: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception in RadixSpline range_query" << std::endl;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / range_queries.size();
    append_to_results_file("RadixSpline", data_type, data_size, "range_search", duration);
    std::cout << "RadixSpline range search: " << duration << " ns/op, found " << total_results << " keys" << std::endl;
}

// Benchmark the insert operation
template <typename T>
void run_insert_benchmark(
    BTree<T> btree,
    BPlusTree<T> bplus_tree,
    RMI<T> rmi,
    FitingTree<T> fiting_tree,
    RadixSpline<T> radix_spline,
    const std::vector<T>& new_keys,
    const std::string& data_type,
    size_t data_size
) {
    std::cout << "Running insert benchmark for " << data_type << " with " << data_size << " elements..." << std::endl;
    
    // Benchmark B-Tree
    auto start_time = std::chrono::high_resolution_clock::now();
    for (const auto& key : new_keys) {
        btree.insert(key);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / new_keys.size();
    append_to_results_file("B-Tree", data_type, data_size, "insert", duration);
    std::cout << "B-Tree insert: " << duration << " ns/op" << std::endl;
    
    // Benchmark B+-Tree
    start_time = std::chrono::high_resolution_clock::now();
    for (const auto& key : new_keys) {
        bplus_tree.insert(key);
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / new_keys.size();
    append_to_results_file("B+-Tree", data_type, data_size, "insert", duration);
    std::cout << "B+-Tree insert: " << duration << " ns/op" << std::endl;
    
    // append_to_results_file("RMI", data_type, data_size, "insert", -1.0);
    // append_to_results_file("FITing-Tree", data_type, data_size, "insert", -1);
    // append_to_results_file("RadixSpline", data_type, data_size, "insert", -1);
}

// Benchmark the delete operation
template <typename T>
void run_delete_benchmark(
    BTree<T> btree,
    BPlusTree<T> bplus_tree,
    RMI<T> rmi,
    FitingTree<T> fiting_tree,
    RadixSpline<T> radix_spline,
    const std::vector<T>& keys_to_delete,
    const std::string& data_type,
    size_t data_size
) {
    std::cout << "Running delete benchmark for " << data_type << " with " << data_size << " elements..." << std::endl;
    
    // Benchmark B-Tree
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t deleted_count = 0;
    for (const auto& key : keys_to_delete) {
        if (btree.remove(key)) {
            deleted_count++;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / keys_to_delete.size();
    append_to_results_file("B-Tree", data_type, data_size, "delete", duration);
    std::cout << "B-Tree delete: " << duration << " ns/op, deleted " << deleted_count << " keys" << std::endl;
    
    // append_to_results_file("B+-Tree", data_type, data_size, "delete", -1);
    // append_to_results_file("RMI", data_type, data_size, "delete", -1);
    // append_to_results_file("FITing-Tree", data_type, data_size, "delete", -1);
    // append_to_results_file("RadixSpline", data_type, data_size, "delete", -1);
}

// Measure the memory usage of each index structure
template <typename T>
void measure_memory_usage(
    const BTree<T>& btree,
    const BPlusTree<T>& bplus_tree,
    const RMI<T>& rmi,
    const FitingTree<T>& fiting_tree,
    const RadixSpline<T>& radix_spline,
    const std::string& data_type,
    size_t data_size
) {
    std::cout << "Measuring memory usage for " << data_type << " with " << data_size << " elements..." << std::endl;
    
    append_to_results_file("B-Tree", data_type, data_size, "memory", btree.memory_usage());
    append_to_results_file("B+-Tree", data_type, data_size, "memory", bplus_tree.memory_usage());
    append_to_results_file("RMI", data_type, data_size, "memory", rmi.memory_usage());
    append_to_results_file("FITing-Tree", data_type, data_size, "memory", fiting_tree.memory_usage());
    append_to_results_file("RadixSpline", data_type, data_size, "memory", radix_spline.memory_usage());
}
