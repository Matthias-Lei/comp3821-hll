#include "dummy_sketch.hpp"
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mm_file/mm_file.hpp>
#include <string>
#include <vector>

static const std::string kPath = "./data/streams/test_dummy_stream.bin";

static void write_stream(const std::vector<uint64_t>& data)
{
        std::ofstream out(kPath, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(uint64_t)));
}

int main()
{
        write_stream({ 10, 20, 10, 30, 20 });

        mm::file_source<uint64_t> stream(kPath);
        DummySketch sketch;
        for (std::size_t i = 0; i < stream.size(); ++i) {
                sketch.update(stream.data()[i]);
        }

        assert(sketch.estimate() == 3.0);
        std::cout << "test_dummy_sketch passed\n";
        return 0;
}
