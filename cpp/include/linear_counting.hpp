#pragma once
#include "sketch.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

// Linear Counting (Whang et al., 1990). Each hash sets one bit in an m-bit bitmap.
// The estimate is -m ln(V), where V is the fraction of bits still empty.
class LinearCounting : public Sketch {
public:
        explicit LinearCounting(std::size_t m);

        void update(std::uint64_t h) override;
        std::optional<double> estimate() const override;
        std::size_t memory_bits() const override;

        // Bitwise OR of the two bitmaps. Both sketches must have the same m and hash seed.
        void merge(const LinearCounting& other);

        bool operator==(const LinearCounting& other) const;

private:
        std::size_t m_;
        std::vector<std::uint64_t> words_;
};
