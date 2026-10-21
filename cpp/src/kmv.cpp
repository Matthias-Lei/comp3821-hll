#include "kmv.hpp"
#include "hash.hpp"

#include <stdexcept>

#define SEED 0
#define BITS_TO_BYTE 8

KMinValues::KMinValues(size_t k) : k(k) {
    if (k <= 0)
        throw std::invalid_argument("KMinValues: k must be positive");
}

void KMinValues::update(uint64_t x) {
    std::uint64_t v = hash64(x, 0);

    // literally all it does is maintain the smallest k elements
    if (
        kValSet.count(v) == 0
    ) {
        if (kValHeap.size() < k) {
            kValHeap.push(v);
            kValSet.insert(v);
        }
        else if (v < kValHeap.top()) {
            kValSet.insert(v);
            kValHeap.push(v);

            uint64_t top = kValHeap.top();
            kValHeap.pop();
            kValSet.erase(top);
        }
    }
}

std::optional<double> KMinValues::estimate() const {
    // basic estimator (equiv to D^BE_k in the paper)
    double estimateBasic = k / kValHeap.top();

    // unbiased basic estimator (equiv to D^UB_k in the paper)
    double estimateUnbiased = (k - 1) / kValHeap.top();

    return static_cast<double>(estimateUnbiased);
}

std::size_t KMinValues::memory_bits() const {
    return (
        sizeof(kValHeap) * BITS_TO_BYTE
        + sizeof(kValSet) * BITS_TO_BYTE
        + sizeof(size_t) // size of k
    );
}
