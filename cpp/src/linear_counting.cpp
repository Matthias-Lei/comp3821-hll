#include "linear_counting.hpp"
#include <bit>
#include <cmath>
#include <stdexcept>

static constexpr std::size_t kWordBits = 64;

LinearCounting::LinearCounting(std::size_t m)
        : m_(m)
        , words_((m + kWordBits - 1) / kWordBits, 0)
{
        if (m == 0)
                throw std::invalid_argument("LinearCounting: m must be positive");
}

void LinearCounting::update(std::uint64_t h)
{
        const std::size_t bit = h % m_;
        words_[bit / kWordBits] |= std::uint64_t { 1 } << (bit % kWordBits);
}

std::optional<double> LinearCounting::estimate() const
{
        std::size_t set_bits = 0;
        for (std::uint64_t word : words_)
                set_bits += std::popcount(word);

        const std::size_t empty_bits = m_ - set_bits;
        if (empty_bits == 0)
                return std::nullopt;

        const double m = static_cast<double>(m_);
        const double v = static_cast<double>(empty_bits) / m;
        return -m * std::log(v);
}

std::size_t LinearCounting::memory_bits() const
{
        return m_;
}

void LinearCounting::merge(const LinearCounting& other)
{
        if (other.m_ != m_)
                throw std::invalid_argument("LinearCounting: cannot merge sketches of different m");
        for (std::size_t i = 0; i < words_.size(); ++i)
                words_[i] |= other.words_[i];
}

bool LinearCounting::operator==(const LinearCounting& other) const
{
        return m_ == other.m_ && words_ == other.words_;
}
