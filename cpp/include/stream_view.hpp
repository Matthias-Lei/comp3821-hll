#pragma once
#include <cstddef>
#include <cstdint>
#include <span>

// Non-owning view over a stream of 64-bit hashed values.
// Backed by mmap'd Arrow buffer in production; plain array in tests.
using StreamView = std::span<const uint64_t>;
