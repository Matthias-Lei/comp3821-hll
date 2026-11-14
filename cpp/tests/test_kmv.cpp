#include "cebs_stream.hpp"
#include "hash.hpp"
#include "kmv.hpp"
#include "test_helpers.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

// cebs paths
#define BERNOULLI_DATA "/data/streams/bernoulli_observations_seed7427369_m1000.cebs"
#define NORMAL_DATA "/data/streams/normal_observations_seed7427369_m1000.cebs"
#define BETA_DATA "/data/streams/beta_observations_seed7427369_m1000.cebs"
#define BINOMIAL_DATA "/data/streams/binomial_observations_seed7427369_m1000.cebs"
#define CHI_SQUARED_DATA "/data/streams/chi_squared_observations_seed7427369_m1000.cebs"

// builds a kmv sketch
KMinValues build(
    const std::vector<uint64_t>& data, size_t m = DEFAULT_MEMORY_BITS, int seed = DEFAULT_SEED)
{
        KMinValues sketch(m);
        for (std::uint64_t element : data)
                sketch.update(hash64(element, seed));
        return sketch;
}

class MonteCarloTest : public Test {
public:
        std::string streamPath;

        MonteCarloTest(std::string streamPath, bool detailed)
                : Test(detailed)
                , streamPath(streamPath)
        {
        }

        int n = 1000;
        int batchSize = 30;

        bool run() const override
        {
                if (detailed)
                        std::cout << n << " x " << batchSize << "\n";

                CebsStream stream(streamPath);
                uint64_t trueCardinality = getDistinctElements(stream.data(), stream.size());

                std::vector<double> estimates = { };
                std::vector<double> estimateBatch = { };

                std::vector<KMinValues> sketches = { };
                for (int i = 0; i < n * batchSize; i++) {
                        estimateBatch = { };
                        for (int j = 0; j < batchSize; j++) {
                                KMinValues sketch
                                    = build(arrayToVector(stream.data(), stream.size()),
                                        DEFAULT_MEMORY_BITS, i);
                                std::optional<double> estimate = sketch.estimate();
                                if (estimate)
                                        estimateBatch.push_back(*estimate);
                                i++;
                        }

                        double estimateSum = 0;
                        for (double estimate : estimateBatch)
                                estimateSum += estimate;
                        estimates.push_back(estimateSum / estimateBatch.size());
                }

                double mean = getMean(estimates);
                double std = getStd(estimates, mean);
                double tStat = getTStatistic(estimates, trueCardinality, mean, std);
                double pVal = getPValue(tStat);
                double cohensD = getCohensD(estimates, trueCardinality, mean, std);
                double rmse = getRMSE(estimates, mean);
                if (detailed)
                        std::cout << "mean: " << mean << "\nstd: " << std << "\ntstat: " << tStat
                                  << "\npval: " << pVal << "\ncohens d: " << cohensD << "\nrmse: " << rmse << "\n\n";

                return pVal > 0.05;
        }
        std::string failureMsg() const override
        {
                return "monte carlo test failed";
        }
};

// note: must use where k > cardinality
class ExactTest : public Test {
public:
        std::string streamPath;
        int k;
        ExactTest(std::string streamPath, bool detailed, int k = -1)
                : Test(detailed)
                , streamPath(streamPath)
                , k(k)
        {
        }

        bool run() const override
        {
                CebsStream stream(streamPath);
                uint64_t trueCardinality = getDistinctElements(stream.data(), stream.size());

                KMinValues sketch = build(arrayToVector(stream.data(), stream.size()),
                    k == -1 ? (trueCardinality + 1) * 64 : k);
                if (detailed && sketch.estimate()) {
                        std::cout << "estimate: " << *sketch.estimate() << std::endl;
                        std::cout << "true cardinality: " << trueCardinality << std::endl;
                }
                return sketch.estimate() == trueCardinality;
        }

        std::string failureMsg() const override
        {
                return "exact test failed";
        }
};

class MergeTest : public Test {
public:
        std::string streamPath;
        std::string streamPath2;

        MergeTest(std::string streamPath, std::string streamPath2, bool detailed)
                : Test(detailed)
                , streamPath(streamPath)
                , streamPath2(streamPath2)
        {
        }

        int n = 100;
        int batchSize = 30;

        bool run() const override
        {
                CebsStream stream(streamPath);
                CebsStream stream2(streamPath2);

                // get true cardinality
                std::unordered_set<uint64_t> distinct(stream.data(), stream.data() + stream.size());
                std::unordered_set<uint64_t> distinct2(
                    stream2.data(), stream2.data() + stream2.size());
                uint64_t cardinality1 = distinct.size();
                uint64_t cardinality2 = distinct2.size();
                distinct.merge(distinct2);
                uint64_t trueCardinality = distinct.size();

                std::vector<double> estimates = { };
                std::vector<double> estimateBatch = { };

                for (int i = 0; i < n * batchSize; i++) {
                        estimateBatch = { };
                        for (int j = 0; j < batchSize; j++) {
                                KMinValues sketch
                                    = build(arrayToVector(stream.data(), stream.size()),
                                        DEFAULT_MEMORY_BITS, i);
                                KMinValues sketch2
                                    = build(arrayToVector(stream2.data(), stream2.size()),
                                        DEFAULT_MEMORY_BITS, i);
                                sketch.merge(sketch2);

                                std::optional<double> estimate = sketch.estimate();
                                if (estimate)
                                        estimateBatch.push_back(*estimate);
                                i++;
                        }

                        double estimateSum = 0;
                        for (double estimate : estimateBatch)
                                estimateSum += estimate;
                        estimates.push_back(estimateSum / estimateBatch.size());
                }

                double mean = getMean(estimates);
                double std = getStd(estimates, mean);
                double tStat = getTStatistic(estimates, trueCardinality, mean, std);
                double pVal = getPValue(tStat);

                if (detailed) {
                        std::cout << "merged estimate (mean): " << mean << std::endl;
                        std::cout << "std: " << std << std::endl;
                        std::cout << "tStat: " << tStat << std::endl;
                        std::cout << "pval: " << pVal << std::endl;

                        std::cout << "true value: " << trueCardinality << std::endl;
                        std::cout << "direct addition value: " << cardinality1 + cardinality2
                                  << std::endl;
                        std::cout << std::endl;
                }

                // double relError = (*sketch.estimate() - trueCardinality) / trueCardinality;
                // bool passed = -0.1 < relError && relError < 0.1;
                return std == 0 || pVal > 0.05;
        }

        std::string failureMsg() const override
        {
                return "merge test failed";
        }
};

// ===============================================================================================
namespace fs = std::filesystem;
// get location of root directory
fs::path source_file_path(__FILE__);
std::string streamDirectory = source_file_path.parent_path().parent_path().parent_path().string();

MonteCarloTest monteCarloNormal(streamDirectory + NORMAL_DATA, true);
MonteCarloTest monteCarloBinomial(streamDirectory + BINOMIAL_DATA, true);
MonteCarloTest monteCarloBern(streamDirectory + BERNOULLI_DATA, true);
MonteCarloTest monteCarloChi(streamDirectory + CHI_SQUARED_DATA, true);
MonteCarloTest monteCarloBeta(streamDirectory + BETA_DATA, true);

ExactTest exactTestBin(streamDirectory + BINOMIAL_DATA, false);
ExactTest exactTestBinArbK(streamDirectory + BINOMIAL_DATA, false, 1280);
ExactTest exactTestBern(streamDirectory + BERNOULLI_DATA, false);
ExactTest exactTestNorm(streamDirectory + NORMAL_DATA, false);

MergeTest mergeTestNormBern(streamDirectory + NORMAL_DATA, streamDirectory + BERNOULLI_DATA, false);
MergeTest mergeTestNormBin(streamDirectory + NORMAL_DATA, streamDirectory + BINOMIAL_DATA, false);
MergeTest mergeTestNormChi(
    streamDirectory + NORMAL_DATA, streamDirectory + CHI_SQUARED_DATA, false);
MergeTest mergeTestNormBeta(streamDirectory + NORMAL_DATA, streamDirectory + BETA_DATA, false);
MergeTest mergeTestBinBern(
    streamDirectory + BINOMIAL_DATA, streamDirectory + BERNOULLI_DATA, false);

std::vector<TestGroup> testGroups = { TestGroup("Estimate is Unbiased")
                                          .addTest(&monteCarloNormal)
                                          // .addTest(&monteCarloBinomial)
                                          .addTest(&monteCarloChi)
                                          .addTest(&monteCarloBeta)
        // .addTest(&monteCarloBern)
        ,
        TestGroup("Estimate is Exact for n < k")
            .addTest(&exactTestBin)
            .addTest(&exactTestBinArbK)
            .addTest(&exactTestBern)
            .addTest(&exactTestNorm),
        TestGroup("Merging Estimates is Accurate")
            .addTest(&mergeTestNormBern)
            .addTest(&mergeTestNormBin)
            .addTest(&mergeTestNormChi)
            .addTest(&mergeTestNormBeta)
            .addTest(&mergeTestBinBern) };

int main()
{
        std::cout << streamDirectory + NORMAL_DATA << std::endl;
        std::cout << "Current working directory: " << fs::current_path() << '\n';

        for (TestGroup testGroup : testGroups) {
                testGroup.run();
        }

        return 0;
}
