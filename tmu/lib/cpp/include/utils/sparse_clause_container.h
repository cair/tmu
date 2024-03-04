//
// Created by per on 3/2/24.
//

#ifndef TUMLIBPP_SPARSE_CLAUSE_CONTAINER_H
#define TUMLIBPP_SPARSE_CLAUSE_CONTAINER_H
#include <utility>
#include <vector>
#include <unordered_map>
#include <random>
#include <set>
#include <memory>
#include <functional>
#include <iostream>
#include <tl/optional.hpp>


#include "tm_clause_dense.h"

extern "C" {
    #include "fast_rand.h"
}


template<typename ClauseType>
class ValueIterator : public std::iterator<std::forward_iterator_tag, std::shared_ptr<ClauseType>> {
private:
    using MapIterator = typename std::unordered_map<int, std::shared_ptr<ClauseType>>::iterator;
    MapIterator it;

public:
    explicit ValueIterator(MapIterator it) : it(it) {}

    ValueIterator& operator++() { ++it; return *this; } // Prefix increment
    ValueIterator operator++(int) { ValueIterator tmp = *this; ++(*this); return tmp; } // Postfix increment

    bool operator==(const ValueIterator& other) const { return it == other.it; }
    bool operator!=(const ValueIterator& other) const { return it != other.it; }

    std::shared_ptr<ClauseType>& operator*() const { return it->second; }
    std::shared_ptr<ClauseType>* operator->() const { return &it->second; }
};


template<typename Iterator>
class IteratorRange {
private:
    Iterator beginIterator, endIterator;

public:
    IteratorRange(Iterator begin, Iterator end)
            : beginIterator(begin), endIterator(end) {}

    Iterator begin() const { return beginIterator; }
    Iterator end() const { return endIterator; }
};

template<class ClauseType>
class SparseClauseContainer {
    std::mt19937 rng;
    std::vector<int> classes;
    std::unordered_map<int, std::shared_ptr<ClauseType>> d;

public:
    std::shared_ptr<ClauseType> template_instance;


    explicit SparseClauseContainer(unsigned int random_seed)
    : rng(random_seed)
    {
    }

    ~SparseClauseContainer() {
        clear();
    }



    [[nodiscard]] int n_classes() const {
        return classes.size();
    }

    [[nodiscard]] const std::vector<int>& get_classes() const {
        return classes;
    }

    [[nodiscard]] size_t size() const {
        return classes.size();
    }

    void populate(const std::vector<int>& classes_to_populate) {
        for (auto c : classes_to_populate) {
            this->d.emplace(c, std::make_shared<ClauseType>(*template_instance));
            this->classes.push_back(c);
        }
    }

    std::shared_ptr<ClauseType> operator[](int key) {
        try{
            return d.at(key);
        }catch(const std::out_of_range& e){
            std::cerr << "Key not found in SparseClauseContainer" << std::endl;
            std::cerr << e.what() << std::endl;
            std::cerr << "Key: " << key << std::endl;
            std::cerr << "Available keys: ";
            for (auto& item: d) {
                std::cerr << item.first << " ";
            }
            throw e;
        }
    }

    ValueIterator<ClauseType> begin() { return ValueIterator<ClauseType>(d.begin()); }
    ValueIterator<ClauseType> end() { return ValueIterator<ClauseType>(d.end()); }


    tl::optional<int> sample(const std::set<uint32_t>& exclude = {}) {
        // Handle cases where sampling is not possible
        if (n_classes() <= static_cast<int>(exclude.size()) || n_classes() == 1) {
            return tl::nullopt; // Indicates no valid sample found
        }

        while (true) {
            int idx = pcg32_fast() % n_classes();
            int sampled_class = classes[idx];
            if (exclude.find(sampled_class) == exclude.end()) {
                return sampled_class; // Return immediately once a valid class is found
            }
        }
    }




    void insert(int key, std::shared_ptr<ClauseType> value) {
        if (d.find(key) == d.end()) {
            classes.push_back(key);
        }
        d[key] = value;
    }





    void clear() {
        d.clear();
        classes.clear();
    }
};

#endif //TUMLIBPP_SPARSE_CLAUSE_CONTAINER_H
