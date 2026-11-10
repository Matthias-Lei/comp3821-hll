#pragma once

#include "sketch.hpp"
#include <cstdint>
#include <optional>
#include <vector>

class HyperLogLogPlusPlus : public Sketch {
public:
        explicit HyperLogLogPlusPlus(std::size_t budget_bits);

        void update(uint64_t hashed_value) override;
        std::optional<double> estimate() const override;
        std::size_t memory_bits() const override;
        // Allocated vector capacity; excludes object and allocator overhead.
        std::size_t storage_bytes() const;
        void merge(const HyperLogLogPlusPlus& other);

        bool sparse() const
        {
                return sparse_;
        }
        std::size_t register_count() const
        {
                return register_count_;
        }

private:
        void update_sparse(std::uint32_t entry);
        void flush_sparse();
        template <class Function>
        void for_each_sparse(Function&& function) const;
        void update_dense(std::uint64_t hash);
        std::uint8_t register_value(std::size_t index) const;
        void set_register(std::size_t index, std::uint8_t value);
        void to_dense();
        void add_to_dense(std::uint32_t entry);

        unsigned index_bits_;
        std::size_t register_count_;
        std::vector<std::uint8_t> entries_;
        std::vector<std::uint32_t> pending_;
        std::size_t pending_count_ = 0;
        std::vector<std::uint8_t> registers_;
        bool sparse_ = true;
};
