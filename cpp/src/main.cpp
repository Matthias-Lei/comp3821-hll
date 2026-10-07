#include "dummy_sketch.hpp"
#include <cstdint>
#include <iostream>
#include <mm_file/mm_file.hpp>

int main(int argc, char** argv)
{
        if (argc < 2) {
                std::cerr << "usage: " << argv[0] << " <stream.bin>\n";
                return 1;
        }

        mm::file_source<uint64_t> stream(argv[1], mm::advice::sequential);
        const uint64_t* data = stream.data();
        const std::size_t n = stream.size();

        DummySketch sketch;
        for (std::size_t i = 0; i < n; ++i) {
                sketch.update(data[i]);
        }

        std::cout << "elements: " << n << "\n"
                  << "estimate: " << sketch.estimate() << "\n"
                  << "memory:   " << sketch.memory_bits() << " bits\n";
}
