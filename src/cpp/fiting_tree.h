#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <limits>
#include <fstream>
#include "btree.h"

// Structure for storing key-index pairs in B-tree
template <typename KeyType>
struct KeyIndex {
    KeyType key;
    int index;
    
    KeyIndex(KeyType k = 0, int idx = -1) : key(k), index(idx) {}
    
    // Comparison operators for B-tree
    bool operator<(const KeyIndex& other) const { return key < other.key; }
    bool operator>(const KeyIndex& other) const { return key > other.key; }
    bool operator==(const KeyIndex& other) const { return key == other.key; }
    bool operator<=(const KeyIndex& other) const { return key <= other.key; }
    bool operator>=(const KeyIndex& other) const { return key >= other.key; }
};

// Structure for storing linear model segments
template <typename KeyType>
struct Segment {
    KeyType start_key;  // First key covered by the model
    double slope;       // Slope of the linear model (a)
    double intercept;   // Intercept (b)
    int max_error;      // Maximum model error
    int start_position; // Starting position in data array
    int end_position;   // Ending position in data array
};

// Template class FitingTree
template<typename KeyType>
class FitingTree {
private:
    int epsilon;  // Maximum allowable error
    std::vector<Segment<KeyType>> segments;  // Linear model segments
    BTree<KeyIndex<KeyType>, 5> segment_index;  // B-tree for segment indexing
    std::vector<KeyType> data;  // Original data

    // Building segments using "shrinking cone" method
    void buildSegments() {
        segments.clear();
        if (data.empty()) return;
        
        size_t n = data.size();
        size_t start_idx = 0;
        
        while (start_idx < n) {
            // Start with the first point of the current segment
            KeyType start_key = data[start_idx];
            int start_pos = start_idx;
            
            // Extend segment until error exceeds epsilon
            size_t end_idx = start_idx + 1;
            double max_error = 0;
            
            // Initial values for linear regression (least squares method)
            double sum_x = static_cast<double>(data[start_idx]);
            double sum_y = start_pos;
            double sum_xx = static_cast<double>(data[start_idx]) * static_cast<double>(data[start_idx]);
            double sum_xy = static_cast<double>(data[start_idx]) * static_cast<double>(start_pos);
            int count = 1;
            
            double slope = 0;
            double intercept = start_pos;
            
            while (end_idx < n) {
                // Add new point for regression
                count++;
                sum_x += static_cast<double>(data[end_idx]);
                sum_y += end_idx;
                sum_xx += static_cast<double>(data[end_idx]) * static_cast<double>(data[end_idx]);
                sum_xy += static_cast<double>(data[end_idx]) * static_cast<double>(end_idx);
                
                // Calculate slope and intercept using least squares method
                double denominator = count * sum_xx - sum_x * sum_x;
                if (std::abs(denominator) > 1e-10) {
                    slope = (count * sum_xy - sum_x * sum_y) / denominator;
                    intercept = (sum_y - slope * sum_x) / count;
                }
                
                // Check errors for all points from start_idx to end_idx
                double max_current_error = 0;
                for (size_t i = start_idx; i <= end_idx; i++) {
                    double predicted_pos = slope * static_cast<double>(data[i]) + intercept;
                    double error = std::abs(predicted_pos - static_cast<double>(i));
                    max_current_error = std::max(max_current_error, error);
                }
                
                // If error exceeded epsilon, stop
                if (max_current_error > epsilon) {
                    break;
                }
                
                // Otherwise update maximum error and continue
                max_error = max_current_error;
                end_idx++;
            }
            
            // If couldn't extend segment, take only one point
            if (end_idx == start_idx + 1 && max_error > epsilon) {
                slope = 0;
                intercept = start_pos;
                max_error = 0;
                end_idx = start_idx + 1;
            }
            
            // Create and save segment
            Segment<KeyType> segment;
            segment.start_key = start_key;
            segment.slope = slope;
            segment.intercept = intercept;
            segment.max_error = std::ceil(max_error);
            segment.start_position = start_pos;
            segment.end_position = end_idx - 1;
            
            segments.push_back(segment);
            
            // Move to next segment
            start_idx = end_idx;
        }
        
        // Create B-tree for segment indexing
        for (size_t i = 0; i < segments.size(); i++) {
            segment_index.insert(KeyIndex<KeyType>(segments[i].start_key, i));
        }
    }

    // Find segment index for key
    int findSegmentIndex(KeyType key) const {
        // Search for segment with nearest smaller or equal start_key
        auto results = segment_index.range_search(KeyIndex<KeyType>(0), KeyIndex<KeyType>(key));
        
        if (results.empty()) return 0;  // If no suitable segment found, use first one
        
        // Find segment with largest start_key that <= key
        int best_index = results[0].index;
        KeyType best_key = results[0].key;
        
        for (const auto& ki : results) {
            if (ki.key <= key && ki.key >= best_key) {
                best_index = ki.index;
                best_key = ki.key;
            }
        }
        
        return best_index;
    }

    // Structure for insertion buffers (for delta-insertions)
    struct DeltaBuffer {
        std::vector<KeyType> keys;
        int max_size;  // Maximum buffer size
        
        DeltaBuffer(int size = 64) : max_size(size) {}
        
        bool isFull() const {
            return keys.size() >= max_size;
        }
        
        bool insert(KeyType key) {
            if (isFull()) return false;
            
            auto it = std::lower_bound(keys.begin(), keys.end(), key);
            if (it != keys.end() && *it == key) return false;  // Key already exists
            
            keys.insert(it, key);
            return true;
        }
    };
    
    std::vector<DeltaBuffer> delta_buffers;  // Buffers for delta-insertion

public:
    // Constructor
    FitingTree(int epsilon = 32) : epsilon(epsilon) {}
    
    // Load data and build index
    void build(const std::vector<KeyType>& keys) {
        // Copy and sort data
        data = keys;
        std::sort(data.begin(), data.end());
        
        // Build segments
        buildSegments();
        
        // Initialize buffers for delta-insertions
        delta_buffers.resize(segments.size(), DeltaBuffer());
    }
    
    // Key lookup (returns position or -1 if not found)
    int lookup(KeyType key) const {
        if (data.empty() || segments.empty()) {
            return -1;  // Data or model not loaded
        }
        
        // Find suitable segment for key
        int segment_idx = findSegmentIndex(key);
        const Segment<KeyType>& segment = segments[segment_idx];
        
        // Predict position using linear model
        double predicted_pos = segment.slope * static_cast<double>(key) + segment.intercept;
        int pred_pos = static_cast<int>(std::round(predicted_pos));
        
        // Calculate bounds for binary search based on maximum error
        int lower_bound = std::max(segment.start_position, pred_pos - segment.max_error);
        int upper_bound = std::min(segment.end_position, pred_pos + segment.max_error);
        
        // Check that bounds don't exceed array limits
        lower_bound = std::max(0, lower_bound);
        upper_bound = std::min(static_cast<int>(data.size()) - 1, upper_bound);
        
        // Binary search within predicted bounds
        auto it = std::lower_bound(
            data.begin() + lower_bound,
            data.begin() + upper_bound + 1,
            key
        );
        
        if (it != data.end() && *it == key) {
            return std::distance(data.begin(), it);
        }
        
        return -1;  // Key not found
    }
    
    // Range search - returns all keys in range [start, end]
    std::vector<KeyType> range_query(KeyType start, KeyType end) const {
        if (data.empty() || segments.empty() || start > end) {
            return {};
        }
        
        std::vector<KeyType> result;
        
        // Find segment for starting key
        int start_segment_idx = findSegmentIndex(start);
        // Find segment for ending key
        int end_segment_idx = findSegmentIndex(end);
        
        // For each relevant segment
        for (int seg_idx = start_segment_idx; seg_idx <= end_segment_idx && seg_idx < static_cast<int>(segments.size()); ++seg_idx) {
            const Segment<KeyType>& segment = segments[seg_idx];
            
            // Predict position for starting key of this segment
            KeyType search_start = std::max(start, segment.start_key);
            double start_pred_pos = segment.slope * static_cast<double>(search_start) + segment.intercept;
            int start_pos = std::max(segment.start_position, 
                                    static_cast<int>(std::round(start_pred_pos)) - segment.max_error);
            
            // Predict position for ending key of this segment
            KeyType search_end = std::min(end, seg_idx < static_cast<int>(segments.size()) - 1 ? 
                                        segments[seg_idx + 1].start_key - 1 : std::numeric_limits<KeyType>::max());
            double end_pred_pos = segment.slope * static_cast<double>(search_end) + segment.intercept;
            int end_pos = std::min(segment.end_position, 
                                 static_cast<int>(std::round(end_pred_pos)) + segment.max_error);
            
            // Correct bounds for binary search
            start_pos = std::max(0, start_pos);
            end_pos = std::min(static_cast<int>(data.size()) - 1, end_pos);
            
            // Binary search to find starting and ending indices of range
            auto start_it = std::lower_bound(data.begin() + start_pos, data.begin() + end_pos + 1, search_start);
            auto end_it = std::upper_bound(start_it, data.begin() + end_pos + 1, search_end);
            
            // Add all keys from this range
            result.insert(result.end(), start_it, end_it);
        }
        
        return result;
    }
    
    // Key insertion (in-place method)
    bool insertInPlace(KeyType key) {
        if (data.empty() || segments.empty()) {
            data.push_back(key);
            buildSegments();
            return true;
        }
        
        // Find suitable segment for insertion
        int segment_idx = findSegmentIndex(key);
        const Segment<KeyType>& segment = segments[segment_idx];
        
        // Predict insertion position
        double predicted_pos = segment.slope * static_cast<double>(key) + segment.intercept;
        int pred_pos = static_cast<int>(std::round(predicted_pos));
        
        // Find insertion position within error bounds
        int lower_bound = std::max(segment.start_position, pred_pos - segment.max_error);
        int upper_bound = std::min(segment.end_position, pred_pos + segment.max_error);
        
        // Check array bounds
        lower_bound = std::max(0, lower_bound);
        upper_bound = std::min(static_cast<int>(data.size()) - 1, upper_bound);
        
        auto it = std::lower_bound(
            data.begin() + lower_bound,
            data.begin() + upper_bound + 1,
            key
        );
        
        int insert_pos = std::distance(data.begin(), it);
        
        // If key already exists, don't insert
        if (it != data.end() && *it == key) {
            return false;
        }
        
        // Insert key
        data.insert(it, key);
        
        // Update positions in affected segments
        for (auto& s : segments) {
            if (s.start_position >= insert_pos) {
                s.start_position++;
            }
            if (s.end_position >= insert_pos) {
                s.end_position++;
            }
        }
        
        // Check if segments need to be rebuilt
        const Segment<KeyType>& updated_segment = segments[segment_idx];
        if (updated_segment.end_position - updated_segment.start_position > 2 * (segment.end_position - segment.start_position)) {
            buildSegments();
        }
        
        return true;
    }
    
    // Key insertion (delta method)
    bool insertDelta(KeyType key) {
        if (data.empty() || segments.empty()) {
            data.push_back(key);
            buildSegments();
            return true;
        }
        
        // Find suitable segment for insertion
        int segment_idx = findSegmentIndex(key);
        
        // If there's no buffer for this segment yet, create one
        if (static_cast<size_t>(segment_idx) >= delta_buffers.size()) {
            delta_buffers.resize(segment_idx + 1);
        }
        
        // Try to insert key into buffer
        bool inserted = delta_buffers[segment_idx].insert(key);
        if (!inserted) {
            // Buffer is full, merge it with data and retrain model
            std::cout << "Buffer for segment " << segment_idx << " is full, performing merge..." << std::endl;
            
            // Merge with main data
            for (auto& k : delta_buffers[segment_idx].keys) {
                insertInPlace(k);
            }
            delta_buffers[segment_idx].keys.clear();
            
            // Retrain models
            buildSegments();
            
            // Insert new key
            return insertInPlace(key);
        }
        
        // Check size of all buffers
        size_t total_buffer_size = 0;
        for (const auto& buffer : delta_buffers) {
            total_buffer_size += buffer.keys.size();
        }
        
        // If total buffer size exceeded threshold, merge all
        if (total_buffer_size > data.size() * 0.1) { // 10% of main size
            std::cout << "Total buffer size exceeded threshold, performing full merge..." << std::endl;
            
            // Merge all buffers with main data
            for (auto& buffer : delta_buffers) {
                for (auto& k : buffer.keys) {
                    insertInPlace(k);
                }
                buffer.keys.clear();
            }
            
            // Retrain models
            buildSegments();
        }
        
        return true;
    }
    
    // Get number of segments
    size_t segmentCount() const { return segments.size(); }
    
    // Get data size
    size_t dataSize() const { return data.size(); }
    
    // Get data
    const std::vector<KeyType>& getData() const { return data; }

    // Calculate memory usage
    size_t memory_usage() const {
        size_t total_size = 0;
        
        // Size of segments (each segment contains model and range information)
        total_size += segments.size() * sizeof(Segment<KeyType>);
        
        // Size of B-tree index for segment search
        total_size += segment_index.memory_usage();
        
        // Size of stored data
        total_size += data.size() * sizeof(KeyType);
        
        // Size of delta-insertion buffers
        for (const auto& buffer : delta_buffers) {
            total_size += buffer.keys.capacity() * sizeof(KeyType);
            total_size += sizeof(int);  // for max_size
        }
        
        // Size of other class fields
        total_size += sizeof(int);     // epsilon
        total_size += sizeof(bool);    // is_sorted
        
        return total_size;
    }
};
