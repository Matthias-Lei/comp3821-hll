#pragma once
#include <cstdint>
#define XXH_INLINE_ALL
#include "xxhash.h"

inline std::uint64_t hash64(std::uint64_t value, std::uint64_t seed)
{
        unsigned char bytes[8];
        for (int i = 0; i < 8; ++i)
                bytes[i] = static_cast<unsigned char>(value >> (8 * i));
        return XXH64(bytes, sizeof bytes, seed);
}
