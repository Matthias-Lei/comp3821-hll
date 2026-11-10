// Generate raw HLL estimates for synthetic distinct datasets.
// Build: mkdir -p build/experiments
//        c++ -O3 -std=c++20 -Icpp/include -Ibuild/_deps/xxhash-src
//        experiments/hllpp/calibrate.cpp -o build/experiments/hllpp_calibrate
// Usage: hllpp_calibrate <trials> <output.csv>
// To regenerate the checked-in table:
//        build/experiments/hllpp_calibrate 5000 build/experiments/hllpp_raw.csv
//        python3 experiments/hllpp/generate_bias.py <summary.csv> <header.hpp> <raw.csv>
// The checked-in summary is experiments/hllpp/bias_calibration.csv.
// The checked-in header is cpp/src/hyperloglog_plus_plus_bias.hpp.
#include "hash.hpp"

#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {
constexpr std::array<int, 4> kPrecisions { 10, 12, 13, 14 };
constexpr int kPoints = 200;
constexpr std::uint64_t kHashSeed = 1;

std::size_t checkpoint(int p, int point)
{
        const std::size_t max = 5 * (std::size_t { 1 } << p);
        return (max * point + (kPoints - 1) / 2) / (kPoints - 1);
}

class RawHll {
public:
        explicit RawHll(int p)
                : p_(p)
                , registers_(std::size_t { 1 } << p, 0)
        {
                counts_[0] = std::size_t { 1 } << p;
        }

        void update(std::uint64_t hash)
        {
                const auto index = hash >> (64 - p_);
                const auto remaining = (hash << p_) | (std::uint64_t { 1 } << (p_ - 1));
                const auto rank = static_cast<std::uint8_t>(std::countl_zero(remaining) + 1);
                const auto before = registers_[index];
                if (rank > before) {
                        --counts_[before];
                        ++counts_[rank];
                        registers_[index] = rank;
                }
        }

        double estimate() const
        {
                double sum = 0;
                for (std::size_t rank = 0; rank < counts_.size(); ++rank) {
                        sum += std::ldexp(
                            static_cast<double>(counts_[rank]), -static_cast<int>(rank));
                }
                const double m = static_cast<double>(std::size_t { 1 } << p_);
                return (0.7213 / (1.0 + 1.079 / m)) * m * m / sum;
        }

private:
        int p_;
        std::vector<std::uint8_t> registers_;
        std::array<std::size_t, 64> counts_ { };
};
}

int main(int argc, char** argv)
{
        if (argc != 3) {
                std::cerr << "usage: hllpp_calibrate <trials> <output.csv>\n";
                return 1;
        }
        const std::string_view input(argv[1]);
        std::size_t trials = 0;
        const auto [end, error]
            = std::from_chars(input.data(), input.data() + input.size(), trials);
        if (error != std::errc { } || end != input.data() + input.size() || trials == 0) {
                std::cerr << "trials must be a positive integer\n";
                return 1;
        }
        std::ofstream out(argv[2]);
        if (!out) {
                std::cerr << "cannot open output\n";
                return 1;
        }
        out << "trial,p,n,estimate\n" << std::setprecision(17);
        for (std::size_t trial = 0; trial < trials; ++trial) {
                std::mt19937_64 rng(trial);
                std::unordered_set<std::uint64_t> seen;
                seen.reserve(checkpoint(kPrecisions.back(), kPoints - 1));
                std::array<RawHll, 4> sketches { RawHll(10), RawHll(12), RawHll(13), RawHll(14) };
                std::array<int, 4> next { };
                const auto max = checkpoint(kPrecisions.back(), kPoints - 1);
                for (std::size_t n = 0; n <= max; ++n) {
                        if (n != 0) {
                                std::uint64_t value;
                                do {
                                        value = rng();
                                } while (!seen.insert(value).second);
                                const auto hash = hash64(value, kHashSeed);
                                for (std::size_t i = 0; i < sketches.size(); ++i) {
                                        if (next[i] < kPoints) {
                                                sketches[i].update(hash);
                                        }
                                }
                        }
                        for (std::size_t i = 0; i < sketches.size(); ++i) {
                                if (next[i] < kPoints && n == checkpoint(kPrecisions[i], next[i])) {
                                        out << trial << ',' << kPrecisions[i] << ',' << n << ','
                                            << sketches[i].estimate() << '\n';
                                        ++next[i];
                                }
                        }
                }
                if (!out) {
                        std::cerr << "failed to write output\n";
                        return 1;
                }
        }
        out.close();
        if (!out) {
                std::cerr << "failed to close output\n";
                return 1;
        }
}
