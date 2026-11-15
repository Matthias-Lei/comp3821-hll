#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>

// Common interface for all cardinality sketches.
// Maybe change from dynamic to static dispatch if not fast enough
class Sketch {
public:
        // deconstructor, i.e., runs when the object dies
        virtual ~Sketch() = default;

        // i assume it gives the algo another element from the bitstream
        virtual void update(uint64_t x) = 0;


        // return an estimate of the cardinality so far
        // std::nullopt means the sketch cannot give an estimate (e.g. a full bitmap).
        virtual std::optional<double> estimate() const = 0;

        // i assume this returns the bits of memory that the algo (sketch) is using
        virtual std::size_t memory_bits() const = 0;
};
