#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <cmath>

// Generate uniformly distributed random data
std::vector<uint64_t> generate_uniform_data(size_t size) {
    std::vector<uint64_t> data(size);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    
    return data;
}

// Generate normally distributed random data
std::vector<uint64_t> generate_normal_data(size_t size) {
    std::vector<uint64_t> data(size);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::normal_distribution<double> dis(1ULL << 60, 1ULL << 50);
    
    for (size_t i = 0; i < size; ++i) {
        double val = dis(gen);
        if (val < 0) val = 0;
        if (val > std::numeric_limits<uint64_t>::max()) val = std::numeric_limits<uint64_t>::max();
        data[i] = static_cast<uint64_t>(val);
    }
    
    return data;
}

// Generate data following a log-normal distribution
std::vector<uint64_t> generate_lognormal_data(size_t size) {
    std::vector<uint64_t> data(size);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::lognormal_distribution<double> dis(0, 2);
    
    for (size_t i = 0; i < size; ++i) {
        double val = dis(gen) * (1ULL << 60);
        if (val < 0) val = 0;
        if (val > std::numeric_limits<uint64_t>::max()) val = std::numeric_limits<uint64_t>::max();
        data[i] = static_cast<uint64_t>(val);
    }
    
    return data;
}

// Generate data following a Zipf distribution
std::vector<uint64_t> generate_zipf_data(size_t size) {
    std::vector<uint64_t> data(size);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    
    // Zipf parameter (higher means more skew)
    double alpha = 1.5;
    
    // Pre-compute the Zipf distribution
    std::vector<double> zipf_dist(size);
    double sum = 0;
    for (size_t i = 1; i <= size; ++i) {
        zipf_dist[i - 1] = 1.0 / std::pow(i, alpha);
        sum += zipf_dist[i - 1];
    }
    
    // Normalize
    for (size_t i = 0; i < size; ++i) {
        zipf_dist[i] /= sum;
    }
    
    // Generate random data based on the Zipf distribution
    std::uniform_real_distribution<double> dis(0, 1);
    
    for (size_t i = 0; i < size; ++i) {
        double val = dis(gen);
        double cum_prob = 0;
        
        for (size_t j = 0; j < size; ++j) {
            cum_prob += zipf_dist[j];
            if (cum_prob >= val) {
                data[i] = j + 1;
                break;
            }
        }
    }
    
    return data;
}

// Generate data based on the specified distribution
std::vector<uint64_t> generate_data(const std::string& distribution, size_t size) {
    if (distribution == "uniform") {
        return generate_uniform_data(size);
    } else if (distribution == "normal") {
        return generate_normal_data(size);
    } else if (distribution == "lognormal") {
        return generate_lognormal_data(size);
    } else if (distribution == "zipf") {
        return generate_zipf_data(size);
    } else {
        // Default to uniform
        return generate_uniform_data(size);
    }
}

// Generate random keys for search operations
std::vector<uint64_t> generate_search_keys(const std::vector<uint64_t>& data, size_t num_keys) {
    std::vector<uint64_t> search_keys;
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, data.size() - 1);
    
    for (size_t i = 0; i < num_keys; ++i) {
        search_keys.push_back(data[dis(gen)]);
    }
    
    return search_keys;
}
// Generate random range queries
std::vector<std::pair<uint64_t, uint64_t>> generate_range_queries(const std::vector<uint64_t>& data, size_t num_queries) {
    std::vector<std::pair<uint64_t, uint64_t>> range_queries;
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, data.size() - 1);
    
    for (size_t i = 0; i < num_queries; ++i) {
        size_t idx1 = dis(gen);
        size_t idx2 = dis(gen);
        
        if (idx1 > idx2) {
            std::swap(idx1, idx2);
        }
        
        range_queries.push_back({data[idx1], data[idx2]});
    }
    
    return range_queries;
}