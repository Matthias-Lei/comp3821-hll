#pragma once
#include "sketch.hpp"
#include <cstdint>
#include <optional>
#include <vector>

class HyperLogLog : public Sketch {
public:
        static constexpr std::size_t kRegisterBits = 6;

        explicit HyperLogLog(std::size_t budget_bits);

        void update(uint64_t hashed_value) override;
        std::optional<double> estimate() const override;
        std::size_t memory_bits() const override;

        void merge(const HyperLogLog& other);

        std::size_t register_count() const
        {
                return registers_.size();
        }

private:
        unsigned index_bits_;
        std::vector<uint8_t> registers_;
};
