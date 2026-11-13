#include "cebs_stream.hpp"
#include "dummy_sketch.hpp"
#include "test_helpers.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

const std::string kPath = "/tmp/test_dummy_stream.cebs";

void write_cebs(const std::string& path, const std::vector<std::uint64_t>& data)
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

void test_distinct_values(sketch_test::Context& context)
{
        const std::vector<std::uint64_t> values { 10, 20, 10, 30, 20 };
        write_cebs(kPath, values);

        const CebsStream stream(kPath);
        DummySketch sketch;
        for (std::size_t i = 0; i < stream.size(); ++i)
                sketch.update(stream.data()[i]);

        const std::optional<double> estimate = sketch.estimate();
        context.expect(estimate == 3.0, "dummy sketch counts the distinct values");
}

} // namespace

int main(int argc, char** argv)
{
        std::vector<sketch_test::TestGroup> groups;

        groups.emplace_back("Dummy Sketch").add("distinct values", test_distinct_values);

        return sketch_test::run_all(
            argc, argv, std::span<const sketch_test::TestGroup>(groups));
}
