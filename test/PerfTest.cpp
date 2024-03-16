//
// Created by yao on 28/11/18.
//

#include "TestModel.h"
#include "utils_test.h"
#include <boost/format.hpp>
#include "../computeSchur.h"
#include "../GroupModelTypes.h"
#include <chrono>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
using boost::format;

template <typename Traits>
class PerfTest : public TestModel<Traits>{
	RBA_IMPORT_FPTYPES(Traits);
    RBA_TEST_IMPORT_TestModel_MEMBERS(Traits);
public:
    void test(){
        computeHostReference();
		this->setVerbose(true);
        std::chrono::time_point<std::chrono::steady_clock> timePoints[3];
        timePoints[0] = std::chrono::steady_clock::now();
        initializeOptimization();
        timePoints[1] = std::chrono::steady_clock::now();
        optimize(64);
        timePoints[2] = std::chrono::steady_clock::now();
        std::cout << "Time cost:" << std::endl;
        std::cout << format("  Initialization: %f ms.") % std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(timePoints[1] - timePoints[0]).count() << std::endl;
        std::cout << format("  Optimization: %f ms.") % std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(timePoints[2] - timePoints[1]).count() << std::endl;
		double rmsPtErr = 0.f;
		float maxPtErr = 0.f;
		for (uint32_t i = 0; i < this->refPointList.size(); i++) {
			const auto sqrErr = (kmat<lpf, 3>{&this->refPointList.at(i).position[0].value} - kmat<lpf, 3>{&this->pointList.at(i).position[0].value}).sqrNorm();
			EXPECT_LE(sqrErr, 5E-6f);
			rmsPtErr += sqrErr;
			const auto err = std::sqrt(sqrErr);
			if (err > maxPtErr) {
				maxPtErr = err;
			}
		}
		rmsPtErr = std::sqrt(rmsPtErr / this->refPointList.size());
		printf("Point error: rms = %lf, max = %f\n", rmsPtErr, maxPtErr);
		double rmsRErr = 0.f, maxRErr = 0.f;
		double rmsCErr = 0.f, maxCErr = 0.f;
		double rmsVErr = 0.f, maxVErr = 0.f;
		for (uint32_t i = 0; i < this->refCapList.size(); i++) {
			const auto& cap = this->capList.at(i).capture;
			const auto& refCap = this->refCapList.at(i);
			const rba::Pose<lpf>& pose = cap.getPose();
			const rba::Pose<lpf>& refPose = refCap.getPose();
			const Eigen::Quaternionf diffR = Eigen::Quaternionf{refPose.q[0], refPose.q[1], refPose.q[2], refPose.q[3]} * Eigen::Quaternionf{pose.q[0], pose.q[1], pose.q[2], pose.q[3]}.conjugate();
			const float diffTanHalfTheta = std::tan(Eigen::AngleAxisf{diffR}.angle() * 0.5f);
			rmsRErr += diffTanHalfTheta*diffTanHalfTheta;
			if (diffTanHalfTheta > maxRErr) {
				maxRErr = diffTanHalfTheta;
			}
			const auto sqrCErr = (kmat<lpf, 3>{&pose.c[0].value} - kmat<lpf, 3>{&refPose.c[0].value}).sqrNorm();
			rmsCErr += sqrCErr;
			if (sqrCErr > maxCErr * maxCErr) {
				maxCErr = std::sqrt(sqrCErr);
			}
			if constexpr (isRolling(Traits::shutter)) {
				const auto sqrVErr = (cap.getVelocity() - refCap.getVelocity()).sqrNorm();
				rmsVErr += sqrVErr;
				if (sqrVErr > maxVErr * maxVErr) {
					maxVErr = std::sqrt(sqrVErr);
				}
			}
		}
		const auto nbCaps = this->refCapList.size();
		rmsRErr = std::sqrt(rmsRErr / nbCaps);
		rmsCErr = std::sqrt(rmsCErr / nbCaps);
		rmsVErr = std::sqrt(rmsVErr / nbCaps);
		printf("Pose error: rmsR = %lf, maxR = %f, rmsC = %lf, maxC = %lf, rmsV = %lf, maxV = %f\n", rmsRErr, maxRErr, rmsCErr, maxCErr, rmsVErr, maxVErr);
    }
protected:
    void SetUp() override{
#if 1
        nbPts = 3000;
        if (rba::isGroupModel<Traits>()){
            nbIntri = 2;
        }
        nbCaps = 128;
#else
        nbPts = 32;
        if (rba::isGroupModel<Traits>()){
            nbIntri = 1;
        }
        nbCaps = 2;
#endif
        TestModel<Traits>::SetUp();
    }
    void computeHostReference() override{
        // already available in refIntriList/refPointList/refCapList.
    }
};

#define RBA_DEFINE_PerfTest(r, data, TRAITS)\
    using BOOST_PP_CAT(PerfTest_, TRAITS) = PerfTest<rba::TRAITS>;\
    TEST_F(BOOST_PP_CAT(PerfTest_, TRAITS), random)\
    {\
        test();\
    }
BOOST_PP_SEQ_FOR_EACH(RBA_DEFINE_PerfTest, data, ALL_TRAITS)
#undef RBA_DEFINE_PerfTest

