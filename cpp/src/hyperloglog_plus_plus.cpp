#include "hyperloglog_plus_plus.hpp"
#include "hyperloglog_plus_plus_bias.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace {

constexpr unsigned kSparsePrecision = 25;
constexpr unsigned kRankBits = 6;
constexpr std::size_t kBitsPerByte = 8;
constexpr unsigned kSparseRankShift = kRankBits + 1;
constexpr std::uint32_t kRankMask = (1U << kRankBits) - 1;
constexpr std::uint32_t kVarintContinuation = 128;
constexpr std::uint32_t kVarintPayloadMask = kVarintContinuation - 1;
constexpr unsigned kVarintPayloadBits = 7;
constexpr std::uint32_t kBufferHashMultiplier = 2654435761U;

void append_varint(std::vector<std::uint8_t>& output, std::uint32_t value)
{
        while (value >= kVarintContinuation) {
                output.push_back(
                    static_cast<std::uint8_t>((value & kVarintPayloadMask) | kVarintContinuation));
                value >>= kVarintPayloadBits;
        }
        output.push_back(static_cast<std::uint8_t>(value));
}

std::uint32_t read_varint(const std::vector<std::uint8_t>& input, std::size_t& offset)
{
        std::uint32_t result = 0;
        unsigned shift = 0;
        std::uint8_t byte;
        do {
                byte = input[offset++];
                result |= static_cast<std::uint32_t>(byte & kVarintPayloadMask) << shift;
                shift += kVarintPayloadBits;
        } while (byte & kVarintContinuation);
        return result;
}

struct Calibration {
        const BiasPoint* begin;
        const BiasPoint* end;
        double threshold;
};

template <std::size_t N>
Calibration calibration(const BiasPoint (&points)[N], double threshold)
{
        return { points, points + N, threshold };
}

Calibration calibration_for(unsigned p)
{
        switch (p) {
        case 10:
                return calibration(kBiasP10, 900);
        case 12:
                return calibration(kBiasP12, 3100);
        case 13:
                return calibration(kBiasP13, 6500);
        case 14:
                return calibration(kBiasP14, 11500);
        default:
                throw std::invalid_argument(
                    "HyperLogLog++ requires calibrated precision 10, 12, 13 or 14");
        }
}

double estimated_bias(double estimate, Calibration table)
{
        const BiasPoint* upper = std::lower_bound(table.begin, table.end, estimate,
            [](const BiasPoint& point, double value) { return point.raw < value; });
        const BiasPoint* lower = upper;
        double bias_sum = 0;
        for (int i = 0; i < 6; ++i) {
                if (upper == table.end
                    || (lower != table.begin
                        && estimate - (lower - 1)->raw <= upper->raw - estimate)) {
                        bias_sum += (--lower)->bias;
                } else {
                        bias_sum += (upper++)->bias;
                }
        }
        return bias_sum / 6;
}

}

HyperLogLogPlusPlus::HyperLogLogPlusPlus(std::size_t budget_bits)
{
        const std::size_t max_registers = budget_bits / kRankBits;
        if (max_registers < 128) {
                throw std::invalid_argument("HyperLogLog++ budget too small");
        }
        index_bits_ = static_cast<unsigned>(std::bit_width(max_registers) - 1);
        (void)calibration_for(index_bits_);
        register_count_ = std::size_t { 1 } << index_bits_;
}

void HyperLogLogPlusPlus::update(uint64_t hash)
{
        if (!sparse_) {
                update_dense(hash);
                return;
        }
        const auto sparse_index = static_cast<std::uint32_t>(hash >> (64 - kSparsePrecision));
        const unsigned extra_index_bits = kSparsePrecision - index_bits_;
        if ((sparse_index & ((1U << extra_index_bits) - 1)) != 0) {
                update_sparse(sparse_index << 1);
        } else {
                const uint64_t remaining = hash << kSparsePrecision;
                const auto rank = static_cast<std::uint32_t>(
                    remaining == 0 ? 64 - kSparsePrecision + 1 : std::countl_zero(remaining) + 1);
                update_sparse((sparse_index << kSparseRankShift) | (rank << 1) | 1);
        }
}

void HyperLogLogPlusPlus::update_sparse(std::uint32_t entry)
{
        const std::size_t dense_bytes = register_count_ * kRankBits / kBitsPerByte;
        const std::size_t max_slots = std::bit_floor(dense_bytes / (4 * sizeof(std::uint32_t)));
        auto slot_for = [](std::uint32_t value, const std::vector<std::uint32_t>& slots) {
                std::size_t slot = (value * kBufferHashMultiplier) & (slots.size() - 1);
                while (slots[slot] != 0 && slots[slot] != value) {
                        slot = (slot + 1) & (slots.size() - 1);
                }
                return slot;
        };
        if (sparse_buffer_.empty()) {
                sparse_buffer_.assign(std::min(std::size_t { 8 }, max_slots), 0);
        }
        auto slot = slot_for(entry, sparse_buffer_);
        if (sparse_buffer_[slot] == entry) {
                return;
        }
        if (sparse_buffer_count_ + 1 > sparse_buffer_.size() * 3 / 4
            && sparse_buffer_.size() < max_slots) {
                std::vector<std::uint32_t> expanded(
                    std::min(2 * sparse_buffer_.size(), max_slots), 0);
                for (std::uint32_t value : sparse_buffer_) {
                        if (value != 0) {
                                expanded[slot_for(value, expanded)] = value;
                        }
                }
                sparse_buffer_.swap(expanded);
                slot = slot_for(entry, sparse_buffer_);
        }
        sparse_buffer_[slot] = entry;
        ++sparse_buffer_count_;
        if (sparse_buffer_count_ >= max_slots * 3 / 4) {
                flush_sparse();
        }
}

template <class Function>
void HyperLogLogPlusPlus::for_each_sparse(Function&& function) const
{
        std::size_t offset = 0;
        std::vector<std::uint32_t> sorted_buffer;
        sorted_buffer.reserve(sparse_buffer_count_);
        for (std::uint32_t entry : sparse_buffer_) {
                if (entry != 0) {
                        sorted_buffer.push_back(entry);
                }
        }
        std::sort(sorted_buffer.begin(), sorted_buffer.end());
        std::size_t buffer_index = 0;
        std::uint32_t previous_entry = 0;
        std::uint32_t next_compressed_entry = 0;
        bool has_compressed_entry = false;
        std::array<std::uint32_t, 1U << kSparseRankShift> group { };
        std::size_t group_size = 0;

        // Rank-bearing entries for one index can have other entries between them.
        auto emit_group = [&] {
                std::size_t last_ranked_entry = group_size;
                for (std::size_t i = 0; i < group_size; ++i) {
                        if (group[i] & 1) {
                                last_ranked_entry = i;
                        }
                }
                for (std::size_t i = 0; i < group_size; ++i) {
                        if (!(group[i] & 1) || i == last_ranked_entry) {
                                function(group[i]);
                        }
                }
                group_size = 0;
        };

        while (offset < compressed_sparse_.size() || has_compressed_entry
            || buffer_index < sorted_buffer.size()) {
                if (!has_compressed_entry && offset < compressed_sparse_.size()) {
                        previous_entry += read_varint(compressed_sparse_, offset);
                        next_compressed_entry = previous_entry;
                        has_compressed_entry = true;
                }
                std::uint32_t value;
                if (has_compressed_entry
                    && (buffer_index == sorted_buffer.size()
                        || next_compressed_entry <= sorted_buffer[buffer_index])) {
                        value = next_compressed_entry;
                        has_compressed_entry = false;
                } else {
                        value = sorted_buffer[buffer_index++];
                }
                if (group_size > 0
                    && (group[0] >> kSparseRankShift) != (value >> kSparseRankShift)) {
                        emit_group();
                }
                if (group_size == 0 || group[group_size - 1] != value) {
                        group[group_size++] = value;
                }
        }
        emit_group();
}

void HyperLogLogPlusPlus::flush_sparse()
{
        std::vector<std::uint8_t> merged;
        std::uint32_t previous = 0;
        for_each_sparse([&](std::uint32_t entry) {
                append_varint(merged, entry - previous);
                previous = entry;
        });
        if (merged.size() > register_count_ * kRankBits / kBitsPerByte) {
                to_dense();
                return;
        }
        std::vector<std::uint8_t> compact(merged.begin(), merged.end());
        compressed_sparse_.swap(compact);
        std::fill(sparse_buffer_.begin(), sparse_buffer_.end(), 0);
        sparse_buffer_count_ = 0;
}

std::uint8_t HyperLogLogPlusPlus::register_value(std::size_t index) const
{
        const std::size_t bit = index * kRankBits;
        const std::size_t byte = bit / kBitsPerByte;
        const unsigned offset = static_cast<unsigned>(bit % kBitsPerByte);
        std::uint16_t value = dense_registers_[byte];
        if (offset > kBitsPerByte - kRankBits) {
                value |= static_cast<std::uint16_t>(dense_registers_[byte + 1]) << kBitsPerByte;
        }
        return static_cast<std::uint8_t>((value >> offset) & kRankMask);
}

void HyperLogLogPlusPlus::set_register(std::size_t index, std::uint8_t value)
{
        const std::size_t bit = index * kRankBits;
        const std::size_t byte = bit / kBitsPerByte;
        const unsigned offset = static_cast<unsigned>(bit % kBitsPerByte);
        std::uint16_t packed = dense_registers_[byte];
        if (offset > kBitsPerByte - kRankBits) {
                packed |= static_cast<std::uint16_t>(dense_registers_[byte + 1]) << kBitsPerByte;
        }
        packed = static_cast<std::uint16_t>((packed & ~(kRankMask << offset)) | (value << offset));
        dense_registers_[byte] = static_cast<std::uint8_t>(packed);
        if (offset > kBitsPerByte - kRankBits) {
                dense_registers_[byte + 1] = static_cast<std::uint8_t>(packed >> kBitsPerByte);
        }
}

void HyperLogLogPlusPlus::update_dense(std::uint64_t hash)
{
        const std::size_t index = hash >> (64 - index_bits_);
        const std::uint64_t remaining
            = (hash << index_bits_) | (std::uint64_t { 1 } << (index_bits_ - 1));
        const auto rank = static_cast<std::uint8_t>(std::countl_zero(remaining) + 1);
        if (rank > register_value(index)) {
                set_register(index, rank);
        }
}

void HyperLogLogPlusPlus::add_to_dense(std::uint32_t entry)
{
        const bool has_rank = entry & 1;
        const auto index = has_rank ? entry >> kSparseRankShift : entry >> 1;
        uint64_t hash = static_cast<uint64_t>(index) << (64 - kSparsePrecision);
        if (has_rank) {
                const unsigned rank = (entry >> 1) & kRankMask;
                if (rank <= 64 - kSparsePrecision) {
                        hash |= uint64_t { 1 } << (64 - kSparsePrecision - rank);
                }
        }
        update_dense(hash);
}

void HyperLogLogPlusPlus::to_dense()
{
        dense_registers_.assign(register_count_ * kRankBits / kBitsPerByte, 0);
        for_each_sparse([&](std::uint32_t entry) { add_to_dense(entry); });
        std::vector<std::uint8_t>().swap(compressed_sparse_);
        std::vector<std::uint32_t>().swap(sparse_buffer_);
        sparse_buffer_count_ = 0;
        sparse_ = false;
}

std::optional<double> HyperLogLogPlusPlus::estimate() const
{
        if (sparse_) {
                constexpr double m = static_cast<double>(uint64_t { 1 } << kSparsePrecision);
                std::size_t count = 0;
                for_each_sparse([&](std::uint32_t) { ++count; });
                return -m * std::log1p(-static_cast<double>(count) / m);
        }

        double sum = 0;
        std::size_t zeros = 0;
        for (std::size_t index = 0; index < register_count_; ++index) {
                const std::uint8_t value = register_value(index);
                sum += std::ldexp(1.0, -value);
                zeros += value == 0;
        }
        const double m = static_cast<double>(register_count_);
        const Calibration table = calibration_for(index_bits_);
        double estimate = (0.7213 / (1.0 + 1.079 / m)) * m * m / sum;
        if (estimate <= 5 * m) {
                estimate -= estimated_bias(estimate, table);
        }
        if (zeros > 0) {
                const double linear = m * std::log(m / static_cast<double>(zeros));
                if (linear <= table.threshold) {
                        return linear;
                }
        }
        return estimate;
}

std::size_t HyperLogLogPlusPlus::memory_bits() const
{
        return sparse_ ? compressed_sparse_.size() * 8 + sparse_buffer_count_ * 32
                       : register_count_ * kRankBits;
}

std::size_t HyperLogLogPlusPlus::storage_bytes() const
{
        return compressed_sparse_.capacity() + sparse_buffer_.capacity() * sizeof(std::uint32_t)
            + dense_registers_.capacity();
}

void HyperLogLogPlusPlus::merge(const HyperLogLogPlusPlus& other)
{
        if (register_count_ != other.register_count_) {
                throw std::invalid_argument("HyperLogLog++ merge: register counts differ");
        }
        if (this == &other) {
                return;
        }
        if (!other.sparse_) {
                if (sparse_) {
                        to_dense();
                }
                for (std::size_t index = 0; index < register_count_; ++index) {
                        const std::uint8_t value = other.register_value(index);
                        if (value > register_value(index)) {
                                set_register(index, value);
                        }
                }
                return;
        }
        other.for_each_sparse([&](std::uint32_t entry) {
                if (sparse_) {
                        update_sparse(entry);
                } else {
                        add_to_dense(entry);
                }
        });
}
