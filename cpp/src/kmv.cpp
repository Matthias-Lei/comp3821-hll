#include "kmv.hpp"
#include "hash.hpp"

#include <stdexcept>

#define SEED 0
#define BITS_TO_BYTE 8

KMinValues::KMinValues(size_t m) : m(m) {
    if (m <= 0)
        throw std::invalid_argument("KMinValues: m must be positive");

    k = m / 64;
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
    std::uint64_t v = hash64(x, 0);

    tryInsertElement(v);
}

std::optional<double> KMinValues::estimate() const {
    // basic estimator (equiv to D^BE_k in the paper)
    double estimateBasic = k / kValHeap.top();

    // unbiased basic estimator (equiv to D^UB_k in the paper)
    double estimateUnbiased = (k - 1) / kValHeap.top();

    return static_cast<double>(estimateUnbiased);
}

// // number of values to keep (equiv to k in the KMV paper)
// size_t k;

// // list of k smallested hashed values (equiv to L in the KMV paper)
// std::priority_queue<uint64_t> kValHeap;
// std::unordered_set<uint64_t> kValSet;

void KMinValues::merge(KMinValues& other) {
    if (k != other.k) {
        throw std::invalid_argument("KMinValues: cannot merge sketches of different k");
    }

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
        + sizeof(size_t) // size of k
    );
}
