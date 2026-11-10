#include "hash.hpp"
#include "hyperloglog.hpp"
#include "hyperloglog_plus_plus.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {
bool expect(bool condition, const char* message)
{
        if (!condition) {
                std::cerr << "FAILED: " << message << '\n';
        }
        return condition;
}

void add_range(HyperLogLogPlusPlus& sketch, std::uint64_t begin, std::uint64_t end)
{
        for (auto i = begin; i < end; ++i) {
                sketch.update(hash64(i, 7));
        }
}
}

int main()
{
        bool ok = true;
        HyperLogLogPlusPlus empty(8192);
        ok &= expect(*empty.estimate() == 0 && empty.sparse(), "empty sketch");
        ok &= expect(empty.storage_bytes() == 0, "empty sparse sketch has no allocated payload");
        add_range(empty, 0, 10);
        ok &= expect(empty.storage_bytes() < empty.register_count() * 6 / 8,
            "small sparse sketch allocates less payload than dense registers");
        add_range(empty, 0, 100);
        ok &= expect(std::abs(*empty.estimate() - 100) < 0.5, "sparse estimate");
        const auto before = empty.estimate();
        add_range(empty, 0, 100);
        empty.merge(empty);
        ok &= expect(empty.estimate() == before, "duplicates and self merge");

        for (std::size_t n : { 60, 300, 1000, 5000 }) {
                HyperLogLogPlusPlus direct(8192), left(8192), right(8192);
                add_range(direct, 0, n);
                add_range(left, 0, n / 3);
                add_range(right, n / 3, n);
                left.merge(right);
                ok &= expect(
                    left.sparse() == direct.sparse() && left.estimate() == direct.estimate(),
                    "merge equals direct sketch across sparse and dense modes");
                HyperLogLogPlusPlus reverse(8192);
                add_range(reverse, n / 3, n);
                HyperLogLogPlusPlus other(8192);
                add_range(other, 0, n / 3);
                reverse.merge(other);
                ok &= expect(
                    reverse.estimate() == direct.estimate(), "reverse merge equals direct");
        }

        HyperLogLogPlusPlus plus(8192);
        HyperLogLog standard(8192);
        for (std::uint64_t i = 0; i < 100000; ++i) {
                const auto hash = hash64(i, 7);
                plus.update(hash);
                standard.update(hash);
        }
        ok &= expect(!plus.sparse() && plus.estimate() == standard.estimate(),
            "dense estimate matches HLL above 5m");

        HyperLogLogPlusPlus calibrated(8192);
        add_range(calibrated, 1, 3001);
        ok &= expect(std::abs(*calibrated.estimate() - 2914.5785928428104) < 1e-8,
            "dense bias correction agrees with calibrated prototype");

        HyperLogLogPlusPlus collision(8192), collision_direct(8192);
        collision.update(0);
        collision.update(1);
        collision_direct.update(0);
        add_range(collision, 0, 1000);
        add_range(collision_direct, 0, 1000);
        ok &= expect(collision.estimate() == collision_direct.estimate(),
            "sparse prefix collision preserves maximum rank during conversion");

        HyperLogLogPlusPlus interleaved(8192), interleaved_direct(8192);
        const std::uint64_t rare = 32768ULL << 39;
        const std::uint64_t normal = ((1ULL << 21) + 1) << 39;
        interleaved.update(rare | (1ULL << 38));
        interleaved_direct.update(rare);
        interleaved.update(normal);
        interleaved_direct.update(normal);
        for (std::uint64_t i = 1; i <= 100; ++i) {
                interleaved.update(i << 39);
                interleaved_direct.update(i << 39);
        }
        interleaved.update(rare);
        ok &= expect(
            interleaved.sparse() && interleaved.estimate() == interleaved_direct.estimate(),
            "rank variants deduplicate even when normal encodings sort between them");
        HyperLogLogPlusPlus interleaved_merged(8192);
        interleaved_merged.merge(interleaved);
        ok &= expect(interleaved_merged.estimate() == interleaved_direct.estimate(),
            "sparse merge deduplicates interleaved encodings");

        HyperLogLogPlusPlus prefixes(131072);
        std::set<std::uint32_t> distinct_prefixes;
        for (std::uint64_t i = 0; i < 2000; ++i) {
                const std::uint64_t hash = hash64(i / 2, 42);
                prefixes.update(hash);
                distinct_prefixes.insert(static_cast<std::uint32_t>(hash >> 39));
                if (i % 43 == 0) {
                        const double m = static_cast<double>(1ULL << 25);
                        const double expected
                            = -m * std::log1p(-static_cast<double>(distinct_prefixes.size()) / m);
                        ok &= expect(
                            prefixes.sparse() && std::abs(*prefixes.estimate() - expected) < 1e-8,
                            "compressed sparse count agrees with distinct prefixes");
                }
        }

        HyperLogLogPlusPlus rare_before(8192), rare_after(8192);
        HyperLogLog rare_reference(8192);
        rare_before.update(rare);
        rare_reference.update(rare);
        for (std::uint64_t i = 0; i < 1024; ++i) {
                const std::uint64_t hash = (i << 54) | (1ULL << 51);
                rare_before.update(hash);
                rare_after.update(hash);
                rare_reference.update(hash);
        }
        rare_after.update(rare);
        ok &= expect(!rare_before.sparse() && !rare_after.sparse()
                && rare_before.estimate() == rare_after.estimate()
                && rare_before.estimate() == rare_reference.estimate(),
            "rare maximum rank survives compressed sparse conversion");

        HyperLogLogPlusPlus boundary_direct(8192), boundary_left(8192), boundary_right(8192);
        for (std::uint64_t i = 0; i < 269; ++i) {
                const auto hash = hash64(i, 9);
                boundary_direct.update(hash);
                (i % 2 ? boundary_left : boundary_right).update(hash);
        }
        boundary_left.merge(boundary_right);
        ok &= expect(!boundary_direct.sparse() && boundary_left.sparse(),
            "batching can select different representations near conversion");
        for (std::uint64_t i = 269; i < 5000; ++i) {
                const auto hash = hash64(i, 9);
                boundary_direct.update(hash);
                boundary_left.update(hash);
        }
        ok &= expect(
            !boundary_left.sparse() && boundary_direct.estimate() == boundary_left.estimate(),
            "merged and direct sketches agree once both are dense");

        HyperLogLogPlusPlus sparse_first(8192), dense_first(8192);
        const std::uint64_t special[]
            = { 0, 1, 1ULL << 38, 1ULL << 39, 1ULL << 53, (1ULL << 54) | 1, (1ULL << 63) | 1 };
        for (auto hash : special) {
                sparse_first.update(hash);
        }
        for (std::uint64_t i = 1; i <= 192; ++i) {
                dense_first.update(i << 39);
        }
        ok &= expect(dense_first.sparse(), "compressed sparse storage retains 192 entries");
        for (std::uint64_t i = 193; i <= 1200; ++i) {
                dense_first.update(i << 39);
        }
        ok &= expect(!dense_first.sparse(), "sparse storage converts after exceeding dense size");
        ok &= expect(dense_first.storage_bytes() == 1024 * 6 / 8,
            "dense registers use six bits each and release sparse storage");
        for (auto hash : special) {
                dense_first.update(hash);
        }
        for (std::uint64_t i = 1; i <= 1200; ++i) {
                sparse_first.update(i << 39);
        }
        ok &= expect(!sparse_first.sparse() && sparse_first.estimate() == dense_first.estimate(),
            "conversion preserves zero suffixes and index bits");

        bool mismatch = false;
        try {
                plus.merge(HyperLogLogPlusPlus(32768));
        } catch (const std::invalid_argument&) {
                mismatch = true;
        }
        ok &= expect(mismatch, "merge rejects different precisions");
        bool unsupported = false;
        try {
                HyperLogLogPlusPlus uncalibrated(16384);
        } catch (const std::invalid_argument&) {
                unsupported = true;
        }
        ok &= expect(unsupported, "uncalibrated precision is rejected");
        for (std::size_t budget : { 8192, 32768, 65536, 131072 }) {
                HyperLogLogPlusPlus supported(budget);
                add_range(supported, 0, 100);
                ok &= expect(std::abs(*supported.estimate() - 100) < 0.5,
                    "all benchmark budgets are supported");

                const std::size_t count = 4 * supported.register_count();
                HyperLogLogPlusPlus direct(budget), left(budget), right(budget);
                add_range(direct, 0, count);
                add_range(left, 0, count / 2);
                add_range(right, count / 2, count);
                left.merge(right);
                ok &= expect(!direct.sparse() && direct.estimate() == left.estimate(),
                    "conversion and dense merge work at each supported precision");

                HyperLogLog baseline(budget);
                HyperLogLogPlusPlus packed(budget);
                const unsigned p
                    = static_cast<unsigned>(std::bit_width(packed.register_count()) - 1);
                for (std::size_t i = 0; i < packed.register_count(); ++i) {
                        const unsigned rank = 3 + static_cast<unsigned>(i % 4);
                        const auto hash = (static_cast<std::uint64_t>(i) << (64 - p))
                            | (std::uint64_t { 1 } << (64 - p - rank));
                        baseline.update(hash);
                        packed.update(hash);
                }
                ok &= expect(packed.estimate() == baseline.estimate(),
                    "packed dense registers match HLL across byte boundaries");
        }
        return ok ? 0 : 1;
}
