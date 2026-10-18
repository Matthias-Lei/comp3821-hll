#include "dummy_sketch.hpp"

void DummySketch::update(uint64_t x)
{
        seen_.insert(x);
}

std::optional<double> DummySketch::estimate() const
{
        return static_cast<double>(seen_.size());
}

std::size_t DummySketch::memory_bits() const
{
        return seen_.size() * 64;
}
