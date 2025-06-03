#pragma once

#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>

template <typename T, int ORDER = 5>
class BPlusTree {
private:
    struct Node {
        bool is_leaf;
        std::vector<T> keys;
        std::vector<std::shared_ptr<Node>> children;
        std::shared_ptr<Node> next_leaf; // Pointer to the next leaf node (only for leaf nodes)
        
        Node(bool leaf = true) : is_leaf(leaf), next_leaf(nullptr) {}
    };
    
    std::shared_ptr<Node> root;
    size_t size_;
    
    // Split a child node
    void split_child(std::shared_ptr<Node> parent, int index, std::shared_ptr<Node> child) {
        auto new_child = std::make_shared<Node>(child->is_leaf);
        
        // Move the keys and children from the middle+1 to end to the new child
        int mid = ORDER / 2;
        
        for (int i = mid; i < child->keys.size(); ++i) {
            new_child->keys.push_back(child->keys[i]);
        }
        
        if (!child->is_leaf) {
            for (int i = mid; i < child->children.size(); ++i) {
                new_child->children.push_back(child->children[i]);
            }
            
            child->children.resize(mid + 1);
        } else {
            // For leaf nodes, connect the new leaf with the next leaf
            new_child->next_leaf = child->next_leaf;
            child->next_leaf = new_child;
        }
        
        // For non-leaf nodes, the middle key moves up to the parent
        // For leaf nodes, the middle key stays in the leaf
        if (!child->is_leaf) {
            parent->keys.insert(parent->keys.begin() + index, child->keys[mid - 1]);
            child->keys.resize(mid - 1);
        } else {
            parent->keys.insert(parent->keys.begin() + index, new_child->keys[0]);
            child->keys.resize(mid);
        }
        
        // Add the new child to the parent
        parent->children.insert(parent->children.begin() + index + 1, new_child);
    }
    
    // Insert a key into a node that is not full
    void insert_non_full(std::shared_ptr<Node> node, const T& key) {
        int i = node->keys.size() - 1;
        
        if (node->is_leaf) {
            // Insert the key into the leaf node
            node->keys.push_back(key);
            while (i >= 0 && key < node->keys[i]) {
                node->keys[i + 1] = node->keys[i];
                --i;
            }
            node->keys[i + 1] = key;
            ++size_;
        } else {
            // Find the child which is going to have the new key
            while (i >= 0 && key < node->keys[i]) {
                --i;
            }
            ++i;
            
            // If the child is full, split it
            if (node->children[i]->keys.size() == 2 * ORDER - 1) {
                split_child(node, i, node->children[i]);
                
                if (key > node->keys[i]) {
                    ++i;
                }
            }
            
            insert_non_full(node->children[i], key);
        }
    }
    
    // Find the leaf node that should contain the key
    std::shared_ptr<Node> find_leaf(const T& key) const {
        auto node = root;
        
        while (!node->is_leaf) {
            int i = 0;
            while (i < node->keys.size() && key >= node->keys[i]) {
                ++i;
            }
            
            node = node->children[i];
        }
        
        return node;
    }
    
    // Search for a key in a leaf node
    bool search_in_leaf(std::shared_ptr<Node> leaf, const T& key) const {
        return std::binary_search(leaf->keys.begin(), leaf->keys.end(), key);
    }
    
    // Collect all keys in a range [start, end]
    void range_search_internal(const T& start, const T& end, std::vector<T>& result) const {
        auto leaf = find_leaf(start);
        
        // Find all keys in the range in this leaf
        for (const auto& key : leaf->keys) {
            if (key >= start && key <= end) {
                result.push_back(key);
            }
        }
        // Move to the next leaf if needed
        while (leaf->next_leaf && leaf->next_leaf->keys[0] <= end) {
            leaf = leaf->next_leaf;
            
            for (const auto& key : leaf->keys) {
                if (key <= end) {
                    result.push_back(key);
                } else {
                    break; // We've found all keys in the range
                }
            }
        }
    }
    
    // Get the total memory usage of the tree
    size_t memory_usage_internal(const std::shared_ptr<Node>& node) const {
        if (!node) return 0;
        
        size_t memory = sizeof(Node);
        memory += node->keys.capacity() * sizeof(T);
        memory += node->children.capacity() * sizeof(std::shared_ptr<Node>);
        
        for (const auto& child : node->children) {
            memory += memory_usage_internal(child);
        }
        
        return memory;
    }
    
public:
    BPlusTree() : root(std::make_shared<Node>()), size_(0) {}
    
    void insert(const T& key) {
        if (root->keys.size() == 2 * ORDER - 1) {
            // Root is full, create a new root
            auto new_root = std::make_shared<Node>(false);
            new_root->children.push_back(root);
            
            // Split the old root
            split_child(new_root, 0, root);
            
            // Update the root
            root = new_root;
            
            // Insert the key into the appropriate child
            insert_non_full(root, key);
        } else {
            // Root is not full, insert the key
            insert_non_full(root, key);
        }
    }
    
    bool search(const T& key) const {
        auto leaf = find_leaf(key);
        return search_in_leaf(leaf, key);
    }
    
    std::vector<T> range_search(const T& start, const T& end) const {
        std::vector<T> result;
        range_search_internal(start, end, result);
        return result;
    }
    
    size_t size() const {
        return size_;
    }
    
    size_t memory_usage() const {
        try {return memory_usage_internal(root);}
        catch(const std::exception& e) {std::cout << "Memory B+ Tree error" << std::endl;}
    }
    
};