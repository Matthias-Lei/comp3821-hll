#pragma once
#include "sketch.hpp"
#include <optional>
#include <queue>
#include <unordered_set>

// k minimum values
class KMinValues : public Sketch {
public:
    explicit KMinValues(size_t k);

    void update(uint64_t h) override;
    std::optional<double> estimate() const override;
    size_t memory_bits() const override;

    // Bitwise OR of the two bitmaps. Both sketches must have the same m and hash seed.
    void merge(const KMinValues& other);

    bool operator==(const KMinValues& other) const;

private:
    // number of values to keep (equiv to k in the KMV paper)
    size_t k;

    // list of k smallested hashed values (equiv to L in the KMV paper)
    std::priority_queue<uint64_t> kValHeap;
    std::unordered_set<uint64_t> kValSet;
};
