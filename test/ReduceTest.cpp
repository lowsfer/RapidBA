
#include "utils_test.h"
#include <boost/format.hpp>
#include "../GroupModelTypes.h"
#include <random>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
using boost::format;

namespace rba{
template <typename Traits>
cudaError_t launchCudaGetAbsMax(float* absMaxVal, const HessianVec<Traits, true>& vec, cudaStream_t stream);
template <typename Traits>
cudaError_t launchCudaGetSquaredNorm(float* squaredNorm, const HessianVec<Traits, true>& vec, cudaStream_t stream);
}

template <typename Traits>
class ReduceTest : public ::testing::Test{
public:
    RBA_IMPORT_FPTYPES(Traits);
    void testAbsMax(){
        using namespace rba::grouped;
        DevHessianVec<Traits> vec;
        vec.resize(1<<10, 1<<10, 1<<20);
//    vec.resize(15, 0, 0);
        const uint32_t nbTests = 1;
        std::default_random_engine rng;
        std::uniform_real_distribution<float> dist(-32.f, 32.f);
        auto generator = [&]{return dist(rng);};

        for (uint32_t n = 0; n < nbTests; n++){
            float refAbsMaxVal = -INFINITY;
            std::generate(vec.c.begin()->data(), vec.c.end()->data(), generator);
            refAbsMaxVal = std::accumulate(vec.c.begin()->data(), vec.c.end()->data(), refAbsMaxVal,
                                           [](float acc, epf val){ return std::max(acc, std::abs(float(val)));});
            std::generate(vec.a.begin()->data(), vec.a.end()->data(), generator);
            refAbsMaxVal = std::accumulate(vec.a.begin()->data(), vec.a.end()->data(), refAbsMaxVal,
                                           [](float acc, hpf val){ return std::max(acc, std::abs(float(val)));});
            std::generate(vec.b.begin()->data(), vec.b.end()->data(), generator);
            refAbsMaxVal = std::accumulate(vec.b.begin()->data(), vec.b.end()->data(), refAbsMaxVal,
                                           [](float acc, lpf val){ return std::max(acc, std::abs(float(val)));});

            const auto absMaxVal = hostAlloc<float>(1);
            *absMaxVal = 0.f;
            cudaStream_t stream = nullptr;
            checkCudaErrors(rba::launchCudaGetAbsMax(absMaxVal.get(), vec.getParamConst(), stream));
            checkCudaErrors(cudaStreamSynchronize(stream));
            GTEST_ASSERT_EQ(*absMaxVal, refAbsMaxVal);
        }
    }
    void testSquaredNorm(){
        using namespace rba::grouped;
        DevHessianVec<Traits> vec;
        vec.resize(1<<10, 1<<10, 1<<10);
//    vec.resize(1, 0, 0);
        const uint32_t nbTests = 10;
        std::default_random_engine rng;
        std::uniform_real_distribution<float> dist(-32.f, 32.f);
        auto generator = [&]{return dist(rng);};

        for (uint32_t n = 0; n < nbTests; n++){
            double refSqrNorm = 0;
            std::generate(vec.c.begin()->data(), vec.c.end()->data(), generator);
            refSqrNorm = std::accumulate(vec.c.begin()->data(), vec.c.end()->data(), refSqrNorm, [](double acc, epf val){ return acc + sqr(float(val));});
            std::generate(vec.a.begin()->data(), vec.a.end()->data(), generator);
            refSqrNorm = std::accumulate(vec.a.begin()->data(), vec.a.end()->data(), refSqrNorm, [](double acc, hpf val){ return acc + sqr(float(val));});
            std::generate(vec.b.begin()->data(), vec.b.end()->data(), generator);
            refSqrNorm = std::accumulate(vec.b.begin()->data(), vec.b.end()->data(), refSqrNorm, [](double acc, lpf val){ return acc + sqr(float(val));});

            const auto sqrNorm = hostAlloc<float>(1);
            *sqrNorm = 0.f;
            cudaStream_t stream = nullptr;
            checkCudaErrors(rba::launchCudaGetSquaredNorm(sqrNorm.get(), vec.getParamConst(), stream));
            checkCudaErrors(cudaStreamSynchronize(stream));
            GTEST_ASSERT_LE(std::abs(*sqrNorm - refSqrNorm) / refSqrNorm, 1E-2f);
        }
    }
};

#define RBA_DEFINE_ReduceTest(r, data, TRAITS)\
    using BOOST_PP_CAT(ReduceTest_, TRAITS) = ReduceTest<rba::TRAITS>;\
    TEST_F(BOOST_PP_CAT(ReduceTest_, TRAITS), AbsMax)\
    {\
        testAbsMax();\
    }\
    TEST_F(BOOST_PP_CAT(ReduceTest_, TRAITS), SquaredNorm)\
    {\
        testSquaredNorm();\
    }
BOOST_PP_SEQ_FOR_EACH(RBA_DEFINE_ReduceTest, data, ALL_TRAITS)
#undef RBA_DEFINE_ReduceTest

