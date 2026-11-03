#include "kmv.hpp"
#include "hash.hpp"

#include <stdexcept>
#include <iostream>
#include <limits>

#define SEED 0
#define BITS_TO_BYTE 8

KMinValues::KMinValues(size_t m) : m(m) {
    if (m <= 0)
        throw std::invalid_argument("KMinValues: m must be positive");

    k = m / 64;
    // std::cout << "k value is: " << k << std::endl;
}

void KMinValues::tryInsertElement(uint64_t hash) {
    // literally all it does is maintain the smallest k elements
    if (
        kValSet.count(hash) == 0
    ) {
        if (kValHeap.size() < k) {
            kValHeap.push(hash);
            kValSet.insert(hash);
        }
        else if (hash < kValHeap.top()) {
            kValSet.insert(hash);
            kValHeap.push(hash);

            uint64_t top = kValHeap.top();
            kValHeap.pop();
            kValSet.erase(top);
        }
    }
}

void KMinValues::update(uint64_t x) {
    std::uint64_t v = hash64(x, SEED);
    // std::cout << "hashed value: " << v << std::endl;

    tryInsertElement(v);
}

std::optional<double> KMinValues::estimate() const {
    // std::cout << "K is :" << k << std::endl;
    
    // override for n < k
    if (kValSet.size() < k) return kValSet.size();

    // note: multiply by max of uint64 as the hash value has to be normalised
    uint64_t hash = kValHeap.top();

    // basic estimator (equiv to D^BE_k in the paper)
    double estimateBasic = (double)k * std::numeric_limits<uint64_t>::max() / hash;

    // unbiased basic estimator (equiv to D^UB_k in the paper)
    double estimateUnbiased = (double)(k - 1) * std::numeric_limits<uint64_t>::max() / hash;

    // std::cout << "normalised hash: " << (double)hash / std::numeric_limits<uint64_t>::max()  << std::endl;
    return static_cast<double>(estimateUnbiased);
}

// // number of values to keep (equiv to k in the KMV paper)
// size_t k;

// // list of k smallested hashed values (equiv to L in the KMV paper)
// std::priority_queue<uint64_t> kValHeap;
// std::unordered_set<uint64_t> kValSet;

void KMinValues::merge(KMinValues& other) {
    // if (k != other.k) {
    //     throw std::invalid_argument("KMinValues: cannot merge sketches of different k");
    // }

    while (!other.kValHeap.empty()) {
        uint64_t hash = other.kValHeap.top();
        other.kValHeap.pop();

        tryInsertElement(hash);
    }
}

std::size_t KMinValues::memory_bits() const {
    return (
        sizeof(kValHeap) * BITS_TO_BYTE
        + sizeof(kValSet) * BITS_TO_BYTE
        + sizeof(k) * BITS_TO_BYTE // size of k
    );
}
