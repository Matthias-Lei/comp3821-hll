#pragma once
#include "sketch.hpp"
#include <optional>
#include <queue>
#include <unordered_set>

// k minimum values
class KMinValues : public Sketch {
public:
    explicit KMinValues(size_t m);

    void update(uint64_t h) override;
    std::optional<double> estimate() const override;
    size_t memory_bits() const override;

    // Union of the k smallest hashes. Both sketches must have the same k and hash seed.
    void merge(const KMinValues& other);

    bool operator==(const KMinValues& other) const;

    // helper to insert an element already hashed
    void tryInsertElement(uint64_t hash);

private:
    // number of bits of memory avaliable -> affects the k value as well
    size_t m;
    size_t k;

    // list of k smallested hashed values (equiv to L in the KMV paper)
    std::priority_queue<uint64_t> kValHeap;
    std::unordered_set<uint64_t> kValSet;
};
