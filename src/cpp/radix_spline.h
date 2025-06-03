#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>

template <typename KeyType>
class RadixSpline {
private:
    struct SplinePoint {
        KeyType x;
        double y;
    };

    // Storing data inside the class
    std::vector<KeyType> data;

    KeyType min_key_;
    KeyType max_key_;
    size_t num_keys_;

    std::vector<SplinePoint> spline_points_;
    
    size_t num_radix_bits_;
    size_t num_radix_buckets_;
    std::vector<size_t> radix_table_;

    // Getting index in radix table for key
        size_t GetRadixIndex(KeyType key) const {
            // Safe boundary check
            if (key <= min_key_) return 0;
            if (key >= max_key_) return num_radix_buckets_ - 1;
            
            // Safe calculation using normalization
            double normalized_key = static_cast<double>(key - min_key_) / 
                                static_cast<double>(max_key_ - min_key_);
            size_t index = static_cast<size_t>(normalized_key * num_radix_buckets_);
            
            // Additional result check
            return std::min(index, num_radix_buckets_ - 1);
        }

public:
    // Structure for search bounds
    struct SearchBound {
        size_t begin;
        size_t end;
    };

    // Default constructor
    RadixSpline() : min_key_(0), max_key_(0), num_keys_(0), 
                   num_radix_bits_(18), num_radix_buckets_(1ULL << 18) {
        radix_table_.resize(num_radix_buckets_ + 1, 0);
    }

    // Index building method that takes data vector directly
    void build(const std::vector<KeyType>& keys, size_t num_radix_bits = 18) {
        data = keys;
        
        // Parameter initialization
        min_key_ = data.empty() ? 0 : data.front();
        max_key_ = data.empty() ? 0 : data.back();
        num_keys_ = data.size();
        num_radix_bits_ = num_radix_bits;
        num_radix_buckets_ = 1ULL << num_radix_bits_;
        radix_table_.assign(num_radix_buckets_ + 1, 0);

        // Building spline with limited error
        spline_points_.clear();
        spline_points_.push_back({min_key_, 0.0});
        size_t last_pos = 0;

        for (size_t i = 1; i < data.size(); ++i) {
            KeyType key = data[i];
            // Add point if key differs from previous
            if (key != spline_points_.back().x) {
                spline_points_.push_back({key, static_cast<double>(i)});
                last_pos = i;
            }
        }

        if (spline_points_.back().x != max_key_) {
            spline_points_.push_back({max_key_, static_cast<double>(num_keys_ - 1)});
        }

        // Filling radix table
        size_t current_spline_idx = 0;
        for (size_t i = 0; i < num_radix_buckets_; ++i) {
            // Using double to avoid overflow
            double delta = static_cast<double>(max_key_ - min_key_) / num_radix_buckets_;
            KeyType bucket_boundary = min_key_ + static_cast<KeyType>((i + 1) * delta);

            while (current_spline_idx + 1 < spline_points_.size() && 
                   spline_points_[current_spline_idx + 1].x <= bucket_boundary) {
                ++current_spline_idx;
            }
            radix_table_[i] = current_spline_idx;
        }
        radix_table_[num_radix_buckets_] = spline_points_.size() - 1;
    }

    // Getting search bounds for key
    SearchBound get_search_bound(KeyType key) const {
        if (data.empty() || key <= min_key_) return {0, 1};
        if (key >= max_key_) return {num_keys_ - 1, num_keys_};
        if (spline_points_.empty() || spline_points_.size() == 1) {
            return {0, data.size()};
        }

        size_t radix_index = GetRadixIndex(key);
        if (radix_index >= radix_table_.size() - 1) {
            radix_index = radix_table_.size() - 2;
        }
        size_t spline_start = radix_table_[radix_index];
        size_t spline_end = radix_table_[radix_index + 1] + 1;
        if (spline_points_.size() <= 1) return {0, data.size()};  // Protection against empty/single-element spline
        size_t segment_idx = spline_start;
        if (spline_start != spline_end) {
            auto it = std::upper_bound(
                spline_points_.begin() + spline_start,
                spline_points_.begin() + std::min(spline_end, spline_points_.size()),
                key,
                [](KeyType k, const SplinePoint& p) { return k < p.x; }
            );
            if (it == spline_points_.begin() + spline_start) {
                segment_idx = spline_start;  // Protection against initial point
            } else {
                segment_idx = std::distance(spline_points_.begin(), it) - 1;
            }
        }

        if (segment_idx >= spline_points_.size() - 1) {
            segment_idx = (spline_points_.size() >= 2) ? spline_points_.size() - 2 : 0;
        }

        const SplinePoint& p1 = spline_points_[segment_idx];
        const SplinePoint& p2 = spline_points_[segment_idx + 1];

        double dx = static_cast<double>(key - p1.x);
        double dy = p2.y - p1.y;
        double dx_full = static_cast<double>(p2.x - p1.x);

        double pos_estimate = p1.y + (dx * dy) / dx_full;

        double error = std::max(p1.y, p2.y) - std::min(p1.y, p2.y);
        size_t begin = static_cast<size_t>(std::max(0.0, pos_estimate - error));
        size_t end = static_cast<size_t>(std::min(static_cast<double>(num_keys_), pos_estimate + error + 1));

        return {begin, end};
    }

    // Point search
    int lookup(KeyType key) const {
        if (data.empty()) return -1;
        
        SearchBound bound = get_search_bound(key);
        
        // Safe boundary check
        size_t begin = std::min(bound.begin, data.size());
        size_t end = std::min(bound.end, data.size());
        
        // Additional range correctness check
        if (begin >= end || begin >= data.size()) {
            return -1;  // Key not found or boundaries are incorrect
        }
        
        auto start = data.begin() + begin;
        auto stop = data.begin() + end;
        
        auto it = std::lower_bound(start, stop, key);
        if (it != data.end() && *it == key) {
            return std::distance(data.begin(), it);
        }
        return -1;  // Key not found
    }


    // Range search
    std::vector<KeyType> range_query(KeyType start_key, KeyType end_key) const {
        
        std::vector<KeyType> result;
        // std::cerr << "start " << std::endl;
        if (data.empty() || start_key > end_key) return result;

        // std::cerr << "start " << start_key << " Min " << min_key_ << std::endl;

        SearchBound start_bound = get_search_bound(start_key);
        // std::cerr << "Find start_bound " << start_key << std::endl;
        SearchBound end_bound = get_search_bound(end_key);
        // std::cerr << "Find end_bound " << end_key << std::endl;

        size_t begin = start_bound.begin;
        size_t end = std::min(end_bound.end, data.size());
        if (begin >= end || begin >= data.size()) {
            return result;  // Key not found or boundaries are incorrect
        }

        auto start_it = data.begin() + begin;
        auto end_it = data.begin() + end;

        // std::cerr << "Find start_it " << *start_it << std::endl;
        // std::cerr << "Find end_it " << *end_it << std::endl;
        
        for (auto it = std::lower_bound(start_it, end_it, start_key); 
             it != end_it && *it <= end_key; ++it) {
            result.push_back(*it);
        }

        return result;
    }

    // Getting access to data
    const std::vector<KeyType>& get_data() const {
        return data;
    }

    // Data cleanup
    void clear() {
        data.clear();
        spline_points_.clear();
        radix_table_.assign(num_radix_buckets_ + 1, 0);
    }

    // Information methods
    size_t size() const { return data.size(); }
    bool empty() const { return data.empty(); }
    size_t spline_points_size() const { return spline_points_.size(); }
    size_t radix_table_size() const { return radix_table_.size(); }

    size_t memory_usage() const {
        size_t total_size = 0;
        
        // Size of spline points
        total_size += spline_points_.size() * sizeof(SplinePoint);
        
        // Size of radix table
        total_size += radix_table_.size() * sizeof(size_t);
        
        // Size of stored data
        total_size += data.size() * sizeof(KeyType);
        
        // Size of main class fields
        total_size += sizeof(KeyType) * 2;  // min_key_, max_key_
        total_size += sizeof(size_t) * 3;   // num_keys_, num_radix_bits_, num_radix_buckets_        
        return total_size;
    }
};
