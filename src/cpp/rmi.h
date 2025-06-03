#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>

// Class for linear models
class LinearModel {
private:
    double slope;
    double intercept;

public:
    LinearModel(double slope = 0.0, double intercept = 0.0) : slope(slope), intercept(intercept) {}

    double predict(double x) const {
        return slope * x + intercept;
    }
};

// Simple JSON parser for loading models
class SimpleJsonParser {
private:
    std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\n\r");
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(" \t\n\r");
        return str.substr(first, last - first + 1);
    }
    
    double extractNumericValue(const std::string& line, const std::string& key) {
        size_t pos = line.find(key);
        if (pos == std::string::npos) return 0.0;
        
        pos = line.find(':', pos + key.length());
        if (pos == std::string::npos) return 0.0;
        
        std::string valueStr = line.substr(pos + 1);
        valueStr = trim(valueStr);
        
        if (!valueStr.empty() && valueStr.back() == ',') {
            valueStr.pop_back();
        }
        
        return std::stod(valueStr);
    }

public:
    struct ModelParams {
        double slope;
        double intercept;
        int min_error;
        int max_error;
    };
    
    struct RMIModelParams {
        int branch_factor;
        ModelParams stage1;
        std::vector<ModelParams> stage2;
    };
    
    RMIModelParams parseRMIModel(const std::string& filename) {
        RMIModelParams params;
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open model file: " + filename);
        }
        
        std::string line;
        bool inStage1 = false;
        bool inStage2 = false;
        ModelParams currentModel;
        
        while (std::getline(file, line)) {
            line = trim(line);
            
            // Parse branch_factor
            if (line.find("\"branch_factor\"") != std::string::npos) {
                params.branch_factor = static_cast<int>(extractNumericValue(line, "\"branch_factor\""));
            }
            // Parse stage1
            else if (line.find("\"stage1\"") != std::string::npos) {
                inStage1 = true;
                inStage2 = false;
            }
            // Parse stage2
            else if (line.find("\"stage2\"") != std::string::npos) {
                inStage1 = false;
                inStage2 = true;
            }
            else if (inStage1) {
                if (line.find("\"slope\"") != std::string::npos) {
                    params.stage1.slope = extractNumericValue(line, "\"slope\"");
                }
                else if (line.find("\"intercept\"") != std::string::npos) {
                    params.stage1.intercept = extractNumericValue(line, "\"intercept\"");
                }
            }
            else if (inStage2) {
                if (line == "{") {
                    currentModel = ModelParams{};
                }
                else if (line == "}," || line == "}") {
                    params.stage2.push_back(currentModel);
                }
                else if (line.find("\"slope\"") != std::string::npos) {
                    currentModel.slope = extractNumericValue(line, "\"slope\"");
                }
                else if (line.find("\"intercept\"") != std::string::npos) {
                    currentModel.intercept = extractNumericValue(line, "\"intercept\"");
                }
                else if (line.find("\"min_error\"") != std::string::npos) {
                    currentModel.min_error = static_cast<int>(extractNumericValue(line, "\"min_error\""));
                }
                else if (line.find("\"max_error\"") != std::string::npos) {
                    currentModel.max_error = static_cast<int>(extractNumericValue(line, "\"max_error\""));
                }
            }
        }
        
        return params;
    }
};

// Template RMI class
template<typename KeyType>
class RMI {
private:
    int branch_factor;
    LinearModel stage1_model;
    std::vector<LinearModel> stage2_models;
    std::vector<int> min_errors;
    std::vector<int> max_errors;
    
    // Storing data inside the class
    std::vector<KeyType> data;

public:
    RMI() : branch_factor(0) {}
    
    // Load model from JSON file
    bool load_model(const std::string& filename) {
        try {
            SimpleJsonParser parser;
            auto params = parser.parseRMIModel(filename);
            
            branch_factor = params.branch_factor;
            stage1_model = LinearModel(params.stage1.slope, params.stage1.intercept);
            
            stage2_models.clear();
            min_errors.clear();
            max_errors.clear();
            
            for (const auto& model : params.stage2) {
                stage2_models.push_back(LinearModel(model.slope, model.intercept));
                min_errors.push_back(model.min_error);
                max_errors.push_back(model.max_error);
            }
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Error parsing model file: " << e.what() << std::endl;
            return false;
        }
    }

    // Predict position range for key
    std::pair<int, int> predict_position(const KeyType& key) const {
        if (branch_factor == 0) {
            std::cerr << "Error: Model not loaded" << std::endl;
            return {-1, -1};
        }

        // Convert key to double for model
        double key_double = static_cast<double>(key);

        // First level prediction
        double stage1_pred = stage1_model.predict(key_double);
        int model_idx = std::max(0, std::min(branch_factor - 1, static_cast<int>(stage1_pred)));

        // Second level prediction
        double predicted_pos = stage2_models[model_idx].predict(key_double);
        int pred_pos = static_cast<int>(std::round(predicted_pos));

        // Calculate search bounds based on errors
        int lower_bound = std::max(0, pred_pos + min_errors[model_idx]);
        int upper_bound = pred_pos + max_errors[model_idx];

        return {lower_bound, upper_bound};
    }

    // Key search in data
    int lookup(const KeyType& key) const {        
        if (data.empty()) {
            return -1;
        }
        
        if (branch_factor == 0) {
            // If model not loaded, use regular binary search
            auto it = std::lower_bound(data.begin(), data.end(), key);
            if (it != data.end() && *it == key) {
                return std::distance(data.begin(), it);
            }
            return -1;
        }

        auto [lower_bound, upper_bound] = predict_position(key);

        // Check bounds
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
        return -1; // Key not found
    }

    // Range search: finds all keys in range [lower, upper]
    std::vector<KeyType> range_query(const KeyType& lower, const KeyType& upper) const {
        
        if (data.empty() || lower > upper) {
            return {};
        }
        
        std::vector<KeyType> result;
        
        // Find position for lower bound
        typename std::vector<KeyType>::const_iterator lower_it;
        
        if (branch_factor == 0) {
            // If model not loaded, use regular binary search
            lower_it = std::lower_bound(data.begin(), data.end(), lower);
        } else {
            // Use RMI to accelerate lower bound search
            auto [lb_pos, ub_pos] = predict_position(lower);
            lb_pos = std::max(0, lb_pos);
            ub_pos = std::min(static_cast<int>(data.size()) - 1, ub_pos);
            
            lower_it = std::lower_bound(data.begin() + lb_pos, data.begin() + ub_pos + 1, lower);
            if (lower_it == data.begin() + ub_pos + 1) {
                // If not found in predicted bounds, use regular search
                lower_it = std::lower_bound(data.begin(), data.end(), lower);
            }
        }
        
        // If no elements >= lower, return empty result
        if (lower_it == data.end()) {
            return result;
        }
        
        // Find position for upper bound
        typename std::vector<KeyType>::const_iterator upper_it;
        
        if (branch_factor == 0) {
            // Without model use regular search
            upper_it = std::upper_bound(lower_it, data.end(), upper);
        } else {
            // Use RMI to accelerate upper bound search
            auto [lb_pos, ub_pos] = predict_position(upper);
            lb_pos = std::max(static_cast<int>(std::distance(data.begin(), lower_it)), lb_pos);
            ub_pos = std::min(static_cast<int>(data.size()) - 1, ub_pos);
            
            if (lb_pos > ub_pos) {
                // If predicted bounds are incorrect, use regular search
                upper_it = std::upper_bound(lower_it, data.end(), upper);
            } else {
                upper_it = std::upper_bound(data.begin() + lb_pos, data.begin() + ub_pos + 1, upper);
                if (upper_it == data.begin() + lb_pos) {
                    // If not found in predicted bounds, use regular search
                    upper_it = std::upper_bound(lower_it, data.end(), upper);
                }
            }
        }
        
        // Copy all elements from range [lower_it, upper_it) to result
        result.insert(result.end(), lower_it, upper_it);
        
        return result;
    }
    
    // Method for loading data from array
    void load_data(const std::vector<KeyType>& new_data) {
        data = new_data;
    } 
   
   size_t memory_usage() const {
        size_t total_size = 0;
        
        // Size of first level model
        total_size += sizeof(stage1_model);
        
        // Size of all second level models
        total_size += stage2_models.size() * sizeof(LinearModel);
        
        // Size of error arrays
        total_size += min_errors.size() * sizeof(int);
        total_size += max_errors.size() * sizeof(int);
        
        // Size of stored data
        total_size += data.size() * sizeof(KeyType);
        
        // Size of other class fields
        total_size += sizeof(int);      // branch_factor
        total_size += sizeof(size_t);   // num_keys_
        total_size += sizeof(bool);     // is_sorted
        
        return total_size;
    }
};
