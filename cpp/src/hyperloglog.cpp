#include "hyperloglog.hpp"
#include <bit>
#include <cmath>
#include <stdexcept>

namespace {

// Bias correction constant from Flajolet et al. (2007), for m >= 128.
double bias_correction(std::size_t register_count)
{
        return 0.7213 / (1.0 + 1.079 / static_cast<double>(register_count));
}

}

HyperLogLog::HyperLogLog(std::size_t budget_bits)
{
        const std::size_t max_registers = budget_bits / kRegisterBits;
        if (max_registers < 128) {
                throw std::invalid_argument(
                    "HyperLogLog budget too small: need at least 128 registers");
        }

        index_bits_ = 0;
        while ((std::size_t { 2 } << index_bits_) <= max_registers) {
                ++index_bits_;
        }
        registers_.assign(std::size_t { 1 } << index_bits_, 0);
}

void HyperLogLog::update(uint64_t hashed_value)
{
        const std::size_t register_index = hashed_value >> (64 - index_bits_);
        const uint64_t remaining_bits
            = (hashed_value << index_bits_) | (uint64_t { 1 } << (index_bits_ - 1));
        const auto rank = static_cast<uint8_t>(std::countl_zero(remaining_bits) + 1);
        if (rank > registers_[register_index]) {
                registers_[register_index] = rank;
        }
}

std::optional<double> HyperLogLog::estimate() const
{
        const std::size_t register_count = registers_.size();
        double inverse_power_sum = 0.0;
        std::size_t empty_registers = 0;
        for (uint8_t register_value : registers_) {
                inverse_power_sum += std::ldexp(1.0, -register_value);
                if (register_value == 0) {
                        ++empty_registers;
                }
        }

        const double m = static_cast<double>(register_count);
        const double raw_estimate = bias_correction(register_count) * m * m / inverse_power_sum;
        if (raw_estimate <= 2.5 * m && empty_registers > 0) {
                return m * std::log(m / static_cast<double>(empty_registers));
        }
        return raw_estimate;
}

std::size_t HyperLogLog::memory_bits() const
{
        return registers_.size() * kRegisterBits;
}

void HyperLogLog::merge(const HyperLogLog& other)
{
        if (registers_.size() != other.registers_.size()) {
                throw std::invalid_argument("HyperLogLog merge: register counts differ");
        }
        for (std::size_t i = 0; i < registers_.size(); ++i) {
                if (other.registers_[i] > registers_[i]) {
                        registers_[i] = other.registers_[i];
                }
        }
}
