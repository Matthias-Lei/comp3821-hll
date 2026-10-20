#pragma once
#include "sketch.hpp"
#include <unordered_set>

class DummySketch : public Sketch {
public:
        void update(uint64_t x) override;
        std::optional<double> estimate() const override;
        std::size_t memory_bits() const override;

private:
        std::unordered_set<uint64_t> seen_;
};
