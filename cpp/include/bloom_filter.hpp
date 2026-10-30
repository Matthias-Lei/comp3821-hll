#pragma once
#include "sketch.hpp"
#include <cstdint>
#include <optional>
#include <vector>

/**
 * Bloom Filter adapted for cardinality estimation.
 *
 * Estimates cardinality by measuring the fraction of unset (zero)
 * bits remaining in the filter. This generalizes the Linear Counting formula
 * to k hash functions: n ≈ -(m/k) * ln(V/m), where V is the number of empty bits.
 */

class BloomFilter : public Sketch {
public:
        explicit BloomFilter(std::size_t memory_bits);
        void update(uint64_t hashed_value) override;
        std::optional<double> estimate() const override;
        std::size_t memory_bits() const override;

private:
        std::size_t m_;                   // Total bits
        std::size_t k_;                   // Number of hash functions
        std::vector<std::uint64_t> bits_; // Bitmap
};
