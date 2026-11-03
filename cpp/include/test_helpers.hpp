// includes
#include <iostream>
#include <format>
#include <cebs_stream.hpp>
#include <functional>
#include <unordered_set>

// defines
#define DEFAULT_SEED 0
#define DEFAULT_MEMORY_BITS 8192

// ============================================================================
// helper classes for testing sketches
// ============================================================================
// SEE cpp/tests/test_kmv.cpp FOR EXAMPLE USAGE

bool assert(bool check, const char* msg) {
    if (!check) std::cerr << "FAIL: " << msg << '\n';
    return check;
}

// builds a sketch
template <typename SketchType>
SketchType build(const std::vector<uint64_t>& data, size_t m = DEFAULT_MEMORY_BITS, int seed = DEFAULT_SEED) {
    SketchType sketch(m);
    for (std::uint64_t element : data) sketch.update(hash64(element, seed));
    return sketch;
}

// data is the datastream (as an array)
uint64_t getDistinctElements(const uint64_t* data, uint64_t n) {
    std::unordered_set<uint64_t> distinct(data, data + n);
    return distinct.size();
}

// convert c-style array to cpp vector
std::vector<uint64_t> arrayToVector(const std::uint64_t* data, uint64_t n) {
    return std::vector<uint64_t>(data, data + n);
}

// when making a new test, extend this object
class Test {
    public:
    bool detailed;

    // bool detailed - whether to print extra detail for test
    Test(bool detailed = false) : detailed(detailed) {}

    // returns whether test passed
    virtual bool run() const {
        std::cout << "ERROR: Test not implemeneted" << std::endl;
        return false;
    };

    virtual std::string failureMsg() const {
        return "NO FAILURE MSG DEFINED";
    };
};

// for each group of tests, bundle them into here (do NOT extend from here)
// make an instance of this class, add tests to it.
class TestGroup {
    public:
    std::string name;
    std::vector<Test*> tests = {};

    TestGroup(std::string name) : name(name) {}

    // run all the contained tests, return the number of tests failed
    void run(bool detailed = false) {
        int failures = 0;
        for (Test* test : tests) {
            bool testPassed = (*test).run();
            if (!testPassed) {
                failures++;
                std::cout << "test failed | " << (*test).failureMsg() << std::endl;
            }
        }

        std::cout << "Test | " << name << " | " << tests.size() - failures << " of " << tests.size() << " tests succeeded." << std::endl;
    }

    // adds test to test group
    // return self by reference for method chaining
    TestGroup& addTest(Test* newTest) {
        tests.push_back(newTest);
        return *this;
    }
};
