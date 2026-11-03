#include "hash.hpp"
#include "kmv.hpp"
#include "test_helpers.hpp"
#include "cebs_stream.hpp"
#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>

// cebs paths
#define BERNOULLI_DATA "/data/streams/bernoulli_observations_seed7427369.cebs"
#define NORMAL_DATA "/data/streams/normal_observations_seed7427369.cebs"
#define BETA_DATA "/data/streams/beta_observations_seed7427369.cebs"
#define BINOMIAL_DATA "/data/streams/binomial_observations_seed7427369.cebs"
#define CHI_SQUARED_DATA "/data/streams/chi_squared_observations_seed7427369.cebs"


// builds a kmv sketch
KMinValues build(const std::vector<uint64_t>& data, size_t m = DEFAULT_MEMORY_BITS, int seed = DEFAULT_SEED) {
    KMinValues sketch(m);
    for (std::uint64_t element : data) sketch.update(hash64(element, seed));
    return sketch;
}

class MonteCarloTest : public Test {
    public: 
    std::string streamPath;

    MonteCarloTest(std::string streamPath, bool detailed) : Test(detailed), streamPath(streamPath) {}

    int n = 1000000;
    int earlyExitThreshold = 10000;

    bool run() const override {
        CebsStream stream(streamPath);
        uint64_t trueCardinality = getDistinctElements(stream.data(), stream.size());

        double totalBias = 0;
        int estimateCount = 0;
        double biasThreshold = 0.5;

        int earlyExitCountdown = earlyExitThreshold;

        std::vector<KMinValues> sketches = {};
        for (int i = 0; i < n; i++) {
            KMinValues sketch = build(arrayToVector(stream.data(), stream.size()), DEFAULT_MEMORY_BITS, i);
            std::optional<double> estimate = sketch.estimate();
            if (estimate) {
                double bias = *estimate - trueCardinality;
                totalBias += bias;
                estimateCount++;
            }
            if (detailed) {
                if (i % 1000 == 0) std::cout << "current average bias: " << totalBias / estimateCount << std::endl;
            }

            if (-biasThreshold < totalBias / estimateCount && totalBias / estimateCount < biasThreshold ) earlyExitCountdown--;
            else earlyExitCountdown = earlyExitThreshold;

            if (earlyExitCountdown <= 0) {
                std::cout << "reached threshold bias after " << i + 1 << " estimates" << std::endl;
                break;
            }
        }

        double averageBias = totalBias / estimateCount;
        if (detailed) std::cout << "bias: " << averageBias << std::endl;

        bool passed = -0.5 < averageBias && averageBias < 0.5;
        if (passed) std::cout << "monte carlo passed\n";
        else std::cout << "monte carlo failed\n";
        return passed;
    }
    std::string failureMsg() const override {
        return "monte carlo test failed";
    }
};

// note: must use where k > cardinality
class ExactTest : public Test {
    public:
    std::string streamPath;
    int k;
    ExactTest(std::string streamPath, bool detailed, int k = -1) : Test(detailed), streamPath(streamPath), k(k) {}

    bool run() const override {
        CebsStream stream(streamPath);
        uint64_t trueCardinality = getDistinctElements(stream.data(), stream.size());

        KMinValues sketch = build(arrayToVector(stream.data(), stream.size()), k == -1 ? (trueCardinality + 1) * 64 : k);
        if (detailed && sketch.estimate()) {
            std::cout << "estimate: " << *sketch.estimate() << std::endl;
            std::cout << "true cardinality: " << trueCardinality << std::endl;
        }
        return sketch.estimate() == trueCardinality;
    }

    std::string failureMsg() const override {
        return "exact test failed";
    }
};

class MergeTest : public Test {
    public: 
    std::string streamPath;
    std::string streamPath2;

    MergeTest(std::string streamPath, std::string streamPath2, bool detailed)
        : Test(detailed), streamPath(streamPath), streamPath2(streamPath2) {}

    bool run() const override {
        CebsStream stream(streamPath);
        CebsStream stream2(streamPath2);

        // get true cardinality
        std::unordered_set<uint64_t> distinct(stream.data(), stream.data() + stream.size());
        std::unordered_set<uint64_t> distinct2(stream2.data(), stream2.data() + stream2.size());
        uint64_t cardinality1 = distinct.size();
        uint64_t cardinality2 = distinct2.size();
        distinct.merge(distinct2);
        uint64_t trueCardinality = distinct.size();

        KMinValues sketch = build(arrayToVector(stream.data(), stream.size()));
        KMinValues sketch2 = build(arrayToVector(stream2.data(), stream2.size()));
        sketch.merge(sketch2);

        if (detailed) {
            std::cout << "merged estimate: " << *sketch.estimate() << std::endl;
            std::cout << "true value: " << trueCardinality << std::endl;
            std::cout << "direct addition value: " << cardinality1 + cardinality2 << std::endl;
            std::cout << std::endl;
        }
        
        double relError = (*sketch.estimate() - trueCardinality) / trueCardinality;
        bool passed = -0.1 < relError && relError < 0.1;
        return passed;
    }

    std::string failureMsg() const override {
        return "merge test failed";
    }
};


// ===============================================================================================
namespace fs = std::filesystem;
// get location of root directory
fs::path source_file_path(__FILE__);
std::string streamDirectory = source_file_path
    .parent_path()
    .parent_path()
    .parent_path()
    .parent_path()
    .string();

MonteCarloTest monteCarloNormal(streamDirectory + NORMAL_DATA, false);
MonteCarloTest monteCarloBinomial(streamDirectory + BINOMIAL_DATA, false);
MonteCarloTest monteCarloBern(streamDirectory + BERNOULLI_DATA, false);
MonteCarloTest monteCarloChi(streamDirectory + CHI_SQUARED_DATA, false);
MonteCarloTest monteCarloBeta(streamDirectory + BETA_DATA, false);

ExactTest exactTestBin(streamDirectory + BINOMIAL_DATA, false);
ExactTest exactTestBinArbK(streamDirectory + BINOMIAL_DATA, false, 1280);
ExactTest exactTestBern(streamDirectory + BERNOULLI_DATA, false);
ExactTest exactTestNorm(streamDirectory + NORMAL_DATA, false);

MergeTest mergeTestNormBern(streamDirectory + NORMAL_DATA, BERNOULLI_DATA, true);
MergeTest mergeTestNormBin(streamDirectory + NORMAL_DATA, BINOMIAL_DATA, true);
MergeTest mergeTestNormChi(streamDirectory + NORMAL_DATA, CHI_SQUARED_DATA, true);
MergeTest mergeTestNormBeta(streamDirectory + NORMAL_DATA, BETA_DATA, true);
MergeTest mergeTestBinBern(streamDirectory + BINOMIAL_DATA, BERNOULLI_DATA, true);

std::vector<TestGroup> testGroups = {
    TestGroup("Estimate is Unbiased")
        .addTest(&monteCarloNormal)
        .addTest(&monteCarloBinomial)
        .addTest(&monteCarloChi)
        .addTest(&monteCarloBeta)
        .addTest(&monteCarloBern)
    , TestGroup("Estimate is Exact for n < k")
        .addTest(&exactTestBin)
        .addTest(&exactTestBinArbK)
        .addTest(&exactTestBern)
        .addTest(&exactTestNorm)
    , TestGroup("Merging Estimates is Accurate")
        .addTest(&mergeTestNormBern)
        .addTest(&mergeTestNormBin)
        .addTest(&mergeTestNormChi)
        .addTest(&mergeTestNormBeta)
        .addTest(&mergeTestBinBern)
};


int main() {
    std::cout << streamDirectory + NORMAL_DATA << std::endl;
    std::cout << "Current working directory: " << fs::current_path() << '\n';

    for (TestGroup testGroup : testGroups) {
        testGroup.run();
    }

    return 0;
}
