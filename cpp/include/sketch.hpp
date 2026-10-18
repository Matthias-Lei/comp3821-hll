#pragma once
#include <cstddef>
#include <cstdint>

// Common interface for all cardinality sketches.
// Maybe change from dynamic to staic dispatch if not fast enough
class Sketch {
public:
        virtual ~Sketch() = default;
        virtual void update(uint64_t x) = 0;
        virtual double estimate() const = 0;
        virtual std::size_t memory_bits() const = 0;
};
