#include "bloom_filter.hpp"
#include <bit>
#include <cmath>
#include <stdexcept>

static constexpr std::size_t kWordBits = 64;

BloomFilter::BloomFilter(std::size_t memory_bits)
        : m_ { memory_bits }
        , k_ { 3 }
        , bits_((memory_bits + kWordBits - 1) / kWordBits, 0)
{
        if (m_ == 0)
                throw std::invalid_argument("BloomFilter: memory must be positive");
}

void BloomFilter::update(uint64_t hashed_value)
{
        // Kirsch-Mitzenmacher double hashing
        uint32_t h1 = static_cast<uint32_t>(hashed_value);
        uint32_t h2 = static_cast<uint32_t>(hashed_value >> 32);

        for (std::size_t i = 0; i < k_; ++i) {
                std::size_t index = (h1 + i * h2) % m_;
                bits_[index / kWordBits] |= (std::uint64_t { 1 } << (index % kWordBits));
        }
}

std::optional<double> BloomFilter::estimate() const
{
        std::size_t set_bits = 0;
        for (std::uint64_t word : bits_) {
                set_bits += std::popcount(word);
        }
        std::size_t empty_bits = m_ - set_bits;
        if (empty_bits == 0)
                return std::nullopt; // Saturated

        double z = static_cast<double>(empty_bits) / static_cast<double>(m_);
        double estimate = -(static_cast<double>(m_) / static_cast<double>(k_)) * std::log(z);
        return estimate;
}

std::size_t BloomFilter::memory_bits() const
{
        return m_;
}
