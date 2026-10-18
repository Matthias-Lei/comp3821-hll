#include "cebs_stream.hpp"
#include "dummy_sketch.hpp"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static const std::string kPath = "/tmp/test_dummy_stream.cebs";

static void write_cebs(const std::string& path, const std::vector<std::uint64_t>& data)
{
        const std::uint64_t data_start = 128;
        const std::uint64_t stream_type = 0x00; // uint64
        const std::uint64_t bytes_per_element = 8;
        const std::uint64_t number_of_units = data.size();

        std::uint8_t header[data_start] = { };
        std::memcpy(header + 0, "CEBS", 4);
        std::memcpy(header + 4, "a\0\0\0", 4);
        std::memcpy(header + 8, &data_start, 8);
        std::memcpy(header + 16, &stream_type, 8);
        std::memcpy(header + 24, &bytes_per_element, 8);
        std::memcpy(header + 32, &number_of_units, 8);

        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(header), data_start);
        out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(std::uint64_t)));
}

int main()
{
        write_cebs(kPath, { 10, 20, 10, 30, 20 });

        CebsStream stream(kPath);
        DummySketch sketch;
        for (std::size_t i = 0; i < stream.size(); ++i)
                sketch.update(stream.data()[i]);

        if (sketch.estimate() != 3.0) {
                std::cerr << "test failed: expected 3.0, got " << sketch.estimate() << "\n";
                return 1;
        }

        std::cout << "test_dummy_sketch passed\n";
        return 0;
}
