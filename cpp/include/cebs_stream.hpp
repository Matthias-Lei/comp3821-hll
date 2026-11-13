#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mm_file/mm_file.hpp>
#include <stdexcept>
#include <string>

// Reader for CEBS files (see python/cebs_spec.md).
// Expects stream_type = uint64 (0x00) and bytes_per_element = 8.
// Exposes the data section as a const uint64_t*.
class CebsStream {
public:
        static constexpr std::uint64_t kHeaderSize = 128;
        static constexpr std::uint64_t kStreamUint64 = 0x00;
        static constexpr std::uint64_t kStreamInt64 = 0x01;
        static constexpr std::uint64_t kStreamFloat64 = 0x02;
        static constexpr std::uint64_t kStreamXXHash64 = 0x10;

        explicit CebsStream(const std::string& path)
                : mmap_(path, mm::advice::sequential)
        {
                if (mmap_.size() < kHeaderSize)
                        throw std::runtime_error("CEBS: file smaller than header: " + path);

                const std::uint8_t* base = mmap_.data();

                if (std::memcmp(base, "CEBS", 4) != 0)
                        throw std::runtime_error("CEBS: bad magic in " + path);

                data_start_ = read_u64(base + 8);
                const auto stype = read_u64(base + 16);
                bytes_per_element_ = read_u64(base + 24);
                number_of_units_ = read_u64(base + 32);

                if (stype != kStreamUint64 && stype != kStreamInt64 && stype != kStreamFloat64
                    && stype != kStreamXXHash64)
                        throw std::runtime_error("CEBS: expected 64-bit stream, got type "
                            + std::to_string(stype) + " in " + path);
                if (bytes_per_element_ == 0)
                        throw std::runtime_error("CEBS: bytes_per_element is zero in " + path);
                if (data_start_ > mmap_.size())
                        throw std::runtime_error("CEBS: data_start exceeds file size in " + path);

                const std::uint64_t data_bytes = mmap_.size() - data_start_;
                if (number_of_units_ > data_bytes / bytes_per_element_)
                        throw std::runtime_error(
                            "CEBS: declared length exceeds file size in " + path);

                data_ = reinterpret_cast<const std::uint64_t*>(base + data_start_);
                length_ = static_cast<std::size_t>(number_of_units_);
        }

        const std::uint64_t* data() const
        {
                return data_;
        }
        std::size_t size() const
        {
                return length_;
        }
        const std::uint64_t* begin() const
        {
                return data_;
        }
        const std::uint64_t* end() const
        {
                return data_ + length_;
        }

private:
        // CEBS is little-endian by spec; x86-64/ARM64 hosts are little-endian.
        static std::uint64_t read_u64(const std::uint8_t* p)
        {
                std::uint64_t v = 0;
                std::memcpy(&v, p, sizeof(v));
                return v;
        }

        mm::file_source<std::uint8_t> mmap_;
        const std::uint64_t* data_ = nullptr;
        std::size_t length_ = 0;

        std::uint64_t data_start_ = 0;
        std::uint64_t bytes_per_element_ = 0;
        std::uint64_t number_of_units_ = 0;
};
