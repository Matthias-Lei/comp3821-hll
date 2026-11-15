#pragma once

#include "sketch.hpp"
#include <cstdint>
#include <optional>
#include <vector>

class HyperLogLogPlusPlus : public Sketch {
public:
        enum class BiasTableSource {
                PublishedAppendix,
                LocalCalibration,
        };

        explicit HyperLogLogPlusPlus(
            std::size_t budget_bits,
            BiasTableSource bias_table_source = BiasTableSource::PublishedAppendix);

        void update(uint64_t hashed_value) override;
        std::optional<double> estimate() const override;
        std::size_t memory_bits() const override;
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
        void compact_sparse_buffer();
        void flush_sparse();
        template <class Function>
        void for_each_sparse(Function&& function) const;
        void update_dense(std::uint64_t hash);
        void to_dense();
        void add_to_dense(std::uint32_t entry);

        unsigned index_bits_;
        std::size_t register_count_;
        BiasTableSource bias_table_source_;
        std::vector<std::uint8_t> compressed_sparse_;
        std::vector<std::uint32_t> sparse_buffer_;
        std::size_t sparse_buffer_count_ = 0;
        std::vector<std::uint8_t> dense_registers_;
        bool sparse_ = true;
};
