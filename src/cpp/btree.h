#pragma once

#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>

template <typename T, int ORDER = 5>
class BTree {
private:
    struct Node {
        bool is_leaf;
        std::vector<T> keys;
        std::vector<std::shared_ptr<Node>> children;
        
        Node(bool leaf = true) : is_leaf(leaf) {}
    };
    
    std::shared_ptr<Node> root;
    size_t size_;
    
    // Split a child node
    void split_child(std::shared_ptr<Node> parent, int index, std::shared_ptr<Node> child) {
        auto new_child = std::make_shared<Node>(child->is_leaf);
        
        // Move the keys and children from the middle+1 to end to the new child
        int mid = ORDER / 2;
        
        for (int i = mid + 1; i < child->keys.size(); ++i) {
            new_child->keys.push_back(child->keys[i]);
        }
        
        if (!child->is_leaf) {
            for (int i = mid + 1; i < child->children.size(); ++i) {
                new_child->children.push_back(child->children[i]);
            }
            
            child->children.resize(mid + 1);
        }
        
        child->keys.resize(mid);
        
        // Insert the middle key into the parent
        parent->keys.insert(parent->keys.begin() + index, child->keys[mid]);
        
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
    
    // Search for a key in the tree
    bool search_internal(std::shared_ptr<Node> node, const T& key) const {
        int i = 0;
        while (i < node->keys.size() && key > node->keys[i]) {
            ++i;
        }
        
        if (i < node->keys.size() && key == node->keys[i]) {
            return true;
        }
        
        if (node->is_leaf) {
            return false;
        }
        
        return search_internal(node->children[i], key);
    }
    
    // Collect all keys in a range [start, end]
    void range_search_internal(std::shared_ptr<Node> node, const T& start, const T& end, std::vector<T>& result) const {
        if (!node) return;
        
        int i = 0;
        while (i < node->keys.size() && start > node->keys[i]) {
            ++i;
        }
        
        if (!node->is_leaf) {
            range_search_internal(node->children[i], start, end, result);
        }
        
        while (i < node->keys.size() && node->keys[i] <= end) {
            if (node->keys[i] >= start) {
                result.push_back(node->keys[i]);
            }
            
            if (!node->is_leaf && i + 1 < node->children.size()) {
                range_search_internal(node->children[i + 1], start, end, result);
            }
            
            ++i;
        }
    }
    
    // Remove a key from the tree
    bool remove_internal(std::shared_ptr<Node>& node, const T& key) {
        if (!node) return false;
        
        bool found = false;
        int idx = 0;
        
        // Find the key in the current node
        while (idx < node->keys.size() && key > node->keys[idx]) {
            ++idx;
        }
        
        // If the key is found in this node
        if (idx < node->keys.size() && key == node->keys[idx]) {
            found = true;
            
            if (node->is_leaf) {
                // Simply remove the key from the leaf node
                node->keys.erase(node->keys.begin() + idx);
                --size_;
            } else {
                // Replace the key with its successor or predecessor
                if (node->children[idx]->keys.size() >= ORDER) {
                    // Find the predecessor
                    auto current = node->children[idx];
                    while (!current->is_leaf) {
                        current = current->children[current->children.size() - 1];
                    }
                    T predecessor = current->keys[current->keys.size() - 1];
                    
                    node->keys[idx] = predecessor;
                    remove_internal(node->children[idx], predecessor);
                } else if (node->children[idx + 1]->keys.size() >= ORDER) {
                    // Find the successor
                    auto current = node->children[idx + 1];
                    while (!current->is_leaf) {
                        current = current->children[0];
                    }
                    T successor = current->keys[0];
                    
                    node->keys[idx] = successor;
                    remove_internal(node->children[idx + 1], successor);
                } else {
                    // Merge the current key and the child idx + 1 into child idx
                    merge_children(node, idx);
                    remove_internal(node->children[idx], key);
                }
            }
        } else if (!node->is_leaf) {
            // The key is not found in this node, so search in the appropriate child
            bool child_removed = remove_internal(node->children[idx], key);
            
            // If the child has less than the minimum number of keys, fix it
            if (child_removed && node->children[idx]->keys.size() < ORDER - 1) {
                if (idx > 0 && node->children[idx - 1]->keys.size() >= ORDER) {
                    // Borrow a key from the left sibling
                    borrow_from_prev(node, idx);
                } else if (idx < node->children.size() - 1 && node->children[idx + 1]->keys.size() >= ORDER) {
                    // Borrow a key from the right sibling
                    borrow_from_next(node, idx);
                } else if (idx > 0) {
                    // Merge with the left sibling
                    merge_children(node, idx - 1);
                } else {
                    // Merge with the right sibling
                    merge_children(node, idx);
                }
            }
            
            found = child_removed;
        }
        
        // If the root is empty, make its child the new root
        if (node == root && node->keys.size() == 0 && !node->is_leaf) {
            root = node->children[0];
        }
        
        return found;
    }
    
    // Borrow a key from the previous child
    void borrow_from_prev(std::shared_ptr<Node> node, int idx) {
        auto child = node->children[idx];
        auto sibling = node->children[idx - 1];
        
        // Move all keys in child one step ahead
        child->keys.insert(child->keys.begin(), node->keys[idx - 1]);
        
        // Move the last key from sibling to the parent
        node->keys[idx - 1] = sibling->keys.back();
        sibling->keys.pop_back();
        
        if (!child->is_leaf) {
            // Move the last child from sibling to the first position in child
            child->children.insert(child->children.begin(), sibling->children.back());
            sibling->children.pop_back();
        }
    }
    
    // Borrow a key from the next child
    void borrow_from_next(std::shared_ptr<Node> node, int idx) {
        auto child = node->children[idx];
        auto sibling = node->children[idx + 1];
        
        // Move a key from parent to child
        child->keys.push_back(node->keys[idx]);
        
        // Move the first key from sibling to the parent
        node->keys[idx] = sibling->keys.front();
        sibling->keys.erase(sibling->keys.begin());
        
        if (!child->is_leaf) {
            // Move the first child from sibling to the last position in child
            child->children.push_back(sibling->children.front());
            sibling->children.erase(sibling->children.begin());
        }
    }
    
    // Merge two children
    void merge_children(std::shared_ptr<Node> node, int idx) {
        auto child = node->children[idx];
        auto sibling = node->children[idx + 1];
        
        // Add a key from the parent to the child
        child->keys.push_back(node->keys[idx]);
        
        // Move all keys from sibling to child
        child->keys.insert(child->keys.end(), sibling->keys.begin(), sibling->keys.end());
        
        if (!child->is_leaf) {
            // Move all children from sibling to child
            child->children.insert(child->children.end(), sibling->children.begin(), sibling->children.end());
        }
        
        // Remove the key from the parent
        node->keys.erase(node->keys.begin() + idx);
        
        // Remove the sibling from the parent's children
        node->children.erase(node->children.begin() + idx + 1);
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
    BTree() : root(std::make_shared<Node>()), size_(0) {}
    
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
        return search_internal(root, key);
    }
    
    std::vector<T> range_search(const T& start, const T& end) const {
        std::vector<T> result;
        range_search_internal(root, start, end, result);
        return result;
    }
    
    bool remove(const T& key) {
        return remove_internal(root, key);
    }
    
    size_t size() const {
        return size_;
    }
    
    size_t memory_usage() const {
        return memory_usage_internal(root);
    }
};