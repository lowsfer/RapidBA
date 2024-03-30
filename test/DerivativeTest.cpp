/*
Copyright [2024] [Yao Yao]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

//
// Created by yao on 16/09/18.
//

#include <gtest/gtest.h>
#include "../derivative.h"
#include <eigen3/Eigen/Dense>
#include <random>
#include "utils_test.h"
#include "TestModel.h"
#include <boost/format.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
using boost::format;
using namespace rba;

static const int defaultNbTests = 4;

template <typename Variable, typename Func, typename ftype>
auto computeRefDerivative(const Variable& v0, Func&& func, const std::array<ftype, Variable::DoF>& delta) -> kmat<ftype, std::result_of<Func(Variable)>::type::rows(), Variable::DoF>
{
    for (ftype e : delta) {
        assert(e != 0); unused(e);
    }
    kmat<ftype, std::result_of<Func(Variable)>::type::rows(), Variable::DoF> ret = kmat<ftype, std::result_of<Func(Variable)>::type::rows(), Variable::DoF>::zeros();
    if constexpr (Variable::DoF != 0) {
        auto result = toEigenMap(ret);
        for (unsigned i = 0; i < Variable::DoF; i++) {
            Variable l = v0;
            Variable r = v0;
            kmat<ftype, Variable::DoF> update = kmat<ftype, Variable::DoF>::zeros();
            update[i] = -delta[i];
            l.update(update);
            update[i] = delta[i];
            r.update(update);
            const auto valL = func(l);
            const auto valR = func(r);
            result.col(i) = (toEigenMap(valR) - toEigenMap(valL)) / (delta[i] * 2);
        }
    }
    return ret;
}

namespace {
constexpr float deltaIntriF = 1E-1f;
constexpr float deltaIntriC = 1E-1f;
constexpr float deltaIntriD = 1E-2f;

template <typename IntrinsicsType>
static std::array<typename IntrinsicsType::ftype, IntrinsicsType::DoF> makeIntrinsicsDelta(){
    std::array<typename IntrinsicsType::ftype, IntrinsicsType::DoF> result;
    for (unsigned i = 0; i < IntrinsicsType::nbFParams; i++){
        result[i] = deltaIntriF;
    }
    for (unsigned i = 0; i < IntrinsicsType::nbCParams; i++){
        result[IntrinsicsType::nbFParams + i] = deltaIntriC;
    }
    for (unsigned i = 0; i < IntrinsicsType::nbDistortParams; i++){
        result[IntrinsicsType::nbFParams + IntrinsicsType::nbCParams + i] = deltaIntriD;
    }
    return result;
};

template <typename ftype = float, uint32_t velocityDoF = 0>
static constexpr std::array<ftype, rba::Pose<ftype>::DoF + velocityDoF> makePoseDelta(){
	const float g = 3E-3f;
	const float c = 1E-2f;
	const float v = 1E-5f; unused(v);
	if constexpr (velocityDoF == 0) {
    	return {{g, g, g, c, c, c}};
	}
	else if constexpr (velocityDoF == 1) {
    	return {{g, g, g, c, c, c, v}};
	}
	else if constexpr (velocityDoF == 3) {
    	return {{g, g, g, c, c, c, v, v, v}};
	}
};

template <typename CameraType>
static std::array<typename CameraType::ftype, CameraType::DoF> makeCameraDelta(){
    return makeIntrinsicsDelta<CameraType>();
}

template <typename CaptureType>
static std::array<typename CaptureType::ftype, CaptureType::DoF> makeCaptureDelta(){
    if constexpr(CaptureType::DoF == Pose<typename CaptureType::ftype>::DoF + CaptureType::Velocity::DoF){
        return makePoseDelta<typename CaptureType::ftype, CaptureType::Velocity::DoF>();
    }
    else{
        const auto poseDelta = makePoseDelta<typename CaptureType::ftype>();
        const auto intriDelta = makeIntrinsicsDelta<typename CaptureType::Intrinsics>();
        std::array<typename CaptureType::ftype, CaptureType::DoF> result;
        std::copy(poseDelta.begin(), poseDelta.end(), result.begin());
        std::copy(intriDelta.begin(), intriDelta.end(), result.begin() + poseDelta.size());
        return result;
    }
}

template <typename IntrinsicsType> struct Delta{
    static std::array<typename IntrinsicsType::ftype, IntrinsicsType::DoF> intrinsics(){
        std::array<typename IntrinsicsType::ftype, IntrinsicsType::DoF> result;
        for (unsigned i = 0; i < IntrinsicsType::nbFParams; i++){
            result[i] = deltaIntriF;
        }
        for (unsigned i = 0; i < IntrinsicsType::nbCParams; i++){
            result[IntrinsicsType::nbFParams + i] = deltaIntriC;
        }
        for (unsigned i = 0; i < IntrinsicsType::nbDistortParams; i++){
            result[IntrinsicsType::nbFParams + IntrinsicsType::nbCParams + i] = deltaIntriD;
        }
        return result;
    };
};

}// unnamed namespace
template <typename Traits>
struct DerivativeTest : public testing::Test {
    RBA_IMPORT_TRAITS(Traits);
    using ftype = lpf;
    std::default_random_engine rng;
    Intrinsics intrinsics;
    Pose<ftype> pose;
    CamIntr camera;
    Capture capture;
    rba::Point<lpf> point;
    kmat<lpf, 2> observation;
    kmat<ftype, 3, 3> R0;
    kmat<Coordinate<lpf>, 3> C0;
    kmat<locf, 3> p0;
    ErrorDerivative<Traits> errDeriv;

    // print parameters for checking with derivative.py
    std::string printParams() const {
        if constexpr (std::is_same<Traits, TraitsGrpF2C2D5Global>::value) {
            return (format("    q0:%f, q1:%f, q2:%f, q3:%f,\n"
                           "    c0:%f, c1:%f, c2:%f,\n"
                           "    fx:%f, fy:%f,\n"
                           "    cx:%f, cy:%f,\n"
                           "    k1:%f, k2:%f, p1:%f, p2:%f, k3:%f,\n"
                           "    x0:%f, x1:%f, x2:%f,\n"
                           "    g0:0,g1:0,g2:0")
                    % pose.q[0] % pose.q[1] % pose.q[2] % pose.q[3]
                    % pose.c[0].template cast<float>() % pose.c[1].template cast<float>() % pose.c[2].template cast<float>()
                    % intrinsics.f[0] % intrinsics.f[1]
                    % intrinsics.c[0] % intrinsics.c[1]
                    % intrinsics.d.k1() % intrinsics.d.k2() % intrinsics.d.p1() % intrinsics.d.p2() % intrinsics.d.k3()
                    % point.position[0].template cast<float>() % point.position[1].template cast<float>() % point.position[2].template cast<float>()).str();
        }
        else{
            std::runtime_error("This is implemented only for TraitsGrpF2C2D5Global");
        }
    }

    void setRandSeed(uint32_t seed = std::random_device{}()){
        rng = std::default_random_engine(seed);
    }

    void setup() {
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        auto genRand = [&](){return dist(rng);};
		static_assert(std::is_same_v<decltype(intrinsics), typename Traits::Intrinsics>);
        intrinsics = RandParamGen<typename Traits::Intrinsics>::make(genRand);
        pose = RandParamGen<Pose<ftype>>::make(genRand);
        if constexpr (isGroupModel<Traits>()){
            capture.intrIdx = 0;
            camera = intrinsics;
        }
        capture.pose = pose;
        if constexpr (!isGroupModel<Traits>() && Capture::DoF > Pose<ftype>::DoF){
            capture.intrinsics = intrinsics;
        }
		if constexpr (isRolling(Traits::shutter)) {
			const auto v = kmat<float, 3>{{genRand(), genRand(), genRand()}} * 1E-4f;
			capture.velocity = Traits::Velocity::make(v);
		}

        R0 = quat2mat(kmat<ftype, 4>{capture.pose.q});
        C0 = kmat<Coordinate<lpf>, 3>(capture.pose.c);
        {
            Eigen::Transform<ftype,3,Eigen::Isometry> eigenPose;
            eigenPose.linear() = toEigenMap(R0);
			auto randWithMinAbs = [&](ftype minAbs){
				assert(minAbs < 0.5f && minAbs >= 0.f);
				const auto v = dist(rng);
				return std::copysign(std::abs(v) * (0.5f - minAbs) / 0.5f + minAbs, v);
			};
			// near-zero x and y in cam coordinate sys caused high relative error, i.e. for d(pt2d.x)/d(r.x) and d(pt2d.y)/d(r.y)
            auto p = C0 + R0.transpose() * kmat<ftype, 3>({randWithMinAbs(0.1f) * 3.f, randWithMinAbs(0.1f) * 3.f, dist(rng) * 1.f + 5.f});;
            point = Point<lpf>{p[0], p[1], p[2]};
        }
        p0 = kmat<locf, 3>(point.position);

		{
			observation = {0,0};
			auto lastObservation = observation;
			while (true)
			{
				observation = computeErrorDerivativeImpl<Traits>(intrinsics, R0, C0, getVelocity<Traits>(capture), getRollingCenter<Traits>(intrinsics), p0, observation).error + observation;
				if ((observation - lastObservation).sqrNorm() < 1E-6f) {
					break;
				}
				lastObservation = observation;
			}
			observation = observation + kmat<lpf, 2>{dist(rng) * 100.f, dist(rng) * 100.f};
		}

        errDeriv = computeErrorDerivativeImpl<Traits>(intrinsics, R0, C0, getVelocity<Traits>(capture), getRollingCenter<Traits>(intrinsics), p0, observation);
    }
    void testIntriDeriv(uint32_t seed = std::random_device{}(), size_t nbTests = defaultNbTests) {
        if constexpr (Intrinsics::DoF != 0) {
            setRandSeed(seed);
            for (unsigned i = 0; i < nbTests; i++) {
                setup();
                const auto intriRefDeriv = computeRefDerivative(
                        intrinsics,
                        [&](const Intrinsics &intri) {
                            return computeErrorDerivativeImpl<Traits>(intri, R0, C0, getVelocity<Traits>(capture), getRollingCenter<Traits>(intrinsics), p0, observation).error;
                        },
                        makeIntrinsicsDelta<Intrinsics>());
                CHECK_KMAT_CLOSE(errDeriv.jacobian.intrinsics(), intriRefDeriv, 1E-1f, seed);
            }
        }
    }

    void testExtriDeriv(uint32_t seed = std::random_device{}(), size_t nbTests = defaultNbTests) {
        setRandSeed(seed);
        for (unsigned i = 0; i < nbTests; i++) {
            setup();
            const auto poseRefDeriv = computeRefDerivative(
                    pose,
                    [&](const Pose<ftype> & p) {
                        const kmat<ftype, 3, 3> R = quat2mat(kmat<ftype, 4>{p.q});
                        const kmat<locf, 3> C = kmat<locf, 3>(p.c);
                        return computeErrorDerivativeImpl<Traits>(intrinsics, R, C, getVelocity<Traits>(capture), getRollingCenter<Traits>(intrinsics), p0,
                                                      observation).error;
                    },
                    makePoseDelta<ftype>());
            CHECK_KMAT_CLOSE(errDeriv.jacobian.pose(), poseRefDeriv, 1.0E-1f, seed);
        }
    }

    void testCamDeriv(uint32_t seed = std::random_device{}(), size_t nbTests = defaultNbTests) {
        if constexpr (CamIntr::DoF != 0) {
            setRandSeed(seed);
            for (unsigned i = 0; i < nbTests; i++) {
                setup();
                const auto camRefDeriv = computeRefDerivative(
                        camera,
                        [&](const CamIntr &cam) {
                            return computeErrorDerivative<Traits>(cam, capture, p0,
                                                                  observation, {false, false, false}).error;
                        },
                        makeCameraDelta<CamIntr>());
                CHECK_KMAT_CLOSE(errDeriv.jacobian.camera, camRefDeriv, 1E-1f, seed);
            }
        }
    }
    void testCapDeriv(uint32_t seed = std::random_device{}(), size_t nbTests = defaultNbTests) {
        setRandSeed(seed);
        for (unsigned i = 0; i < nbTests; i++) {
            setup();
            const auto poseRefDeriv = computeRefDerivative(
                    capture,
                    [&](const Capture& cap) {
                        return computeErrorDerivative<Traits>(camera, cap, p0,
                                                              observation, {false, false, false}).error;
                    },
                    makeCaptureDelta<Capture>());
            CHECK_KMAT_CLOSE(errDeriv.jacobian.capture, poseRefDeriv, 1.0E-1f, seed);
        }
    }

    void testPtDeriv(uint32_t seed = std::random_device{}(), size_t nbTests = defaultNbTests) {
        setRandSeed(seed);
        for (unsigned i = 0; i < nbTests; i++) {
            setup();
            const auto ptRefDeriv = computeRefDerivative(
                    point,
                    [&](const rba::Point<lpf> &pt) {
                        const kmat<locf, 3> p(pt.position);
                        return computeErrorDerivativeImpl<Traits>(intrinsics, R0, C0, getVelocity<Traits>(capture), getRollingCenter<Traits>(intrinsics), p,
                                                      observation).error;
                    },
                    std::array<ftype, 3>{{5E-2f, 5E-2f, 5E-2f}});
            CHECK_KMAT_CLOSE(errDeriv.jacobian.pt, ptRefDeriv, 1E-1f, seed);
        }
    }
};

#define RBA_DEFINE_DerivativeTest(r, data, TRAITS)\
    using BOOST_PP_CAT(DerivativeTest_, TRAITS) = DerivativeTest<rba::TRAITS>;\
    \
    TEST_F(BOOST_PP_CAT(DerivativeTest_, TRAITS), Intrinsics)\
    {\
    	testIntriDeriv(0);\
    }\
    \
    TEST_F(BOOST_PP_CAT(DerivativeTest_, TRAITS), Extrinsics)\
    {\
        testExtriDeriv(0);\
    }\
    \
    TEST_F(BOOST_PP_CAT(DerivativeTest_, TRAITS), Camera)\
    {\
        testCamDeriv(0);\
    }\
    \
    TEST_F(BOOST_PP_CAT(DerivativeTest_, TRAITS), Capture)\
    {\
        testCapDeriv(0);\
    }\
    \
    TEST_F(BOOST_PP_CAT(DerivativeTest_, TRAITS), Point)\
    {\
        testPtDeriv(0);\
    }
BOOST_PP_SEQ_FOR_EACH(RBA_DEFINE_DerivativeTest, data, ALL_TRAITS)
#undef RBA_DEFINE_DerivativeTest

template <typename Traits>
struct IntrinsicsDerivativeTest : DerivativeTest<Traits>
{
    RBA_IMPORT_TRAITS(Traits);
    using typename DerivativeTest<Traits>::ftype;
    kmat<lpf, 3> transXYZ;
    struct NormXY{
        static constexpr uint32_t DoF = 2;
        kmat<lpf, DoF> data;
        void update(const kmat<lpf, DoF>& delta){
            data = data + delta;
        }
    } normXY;
    struct Distort{
        static constexpr uint32_t DoF = Intrinsics::nbDistortParams;
        kmat<lpf, DoF> data;;
        void update(const kmat<lpf, DoF>& delta){
            data = data + delta;
        }
    } distort;
    typename rba::DistortParams<ftype, Intrinsics::nbDistortParams>::ValueJacobian distValJac;

    void setup(){
        DerivativeTest<Traits>::setup();
		auto rollingDelta = kmat<ftype, 3>::zeros();
		if constexpr (isRolling(Traits::shutter)) {
			rollingDelta = getVelocity<Traits>(this->capture).getVelocity() * (this->observation[1] - getRollingCenter<Traits>(this->intrinsics));
		}
        transXYZ = computeTransPointDerivative<Traits::shutter>(this->R0, this->C0, this->p0, rollingDelta).position;
        normXY.data = transXYZ.template block<2,1>(0,0) * fast_rcp(transXYZ[2]);
        distort.data = this->intrinsics.d.params;
        distValJac = this->intrinsics.computeDistortValueJacobian(normXY.data);
    }

    void testNormXY(uint32_t seed = std::random_device{}(), size_t nbTests = defaultNbTests){
        this->setRandSeed(seed);
        for (unsigned i = 0; i < nbTests; i++) {
            setup();
            const auto normXYRefDeriv = computeRefDerivative(
                    normXY,
                    [&](const NormXY &varNormXY) {
                        return this->intrinsics.computeDistortValueJacobian(varNormXY.data).value;
                    },
                    std::array<ftype, 2>{{1E-4f, 1E-4f}});
            CHECK_KMAT_CLOSE(distValJac.normXYJac, normXYRefDeriv, 1E-3f, seed);
        }
    }

    void testDistort(uint32_t seed = std::random_device{}(), size_t nbTests = defaultNbTests){
        this->setRandSeed(seed);
        for (unsigned i = 0; i < nbTests; i++) {
            setup();
            std::array<lpf, Intrinsics::nbDistortParams> delta{};
            for (auto& d : delta) d = 1E-4f;
            const auto distortRefDeriv = computeRefDerivative(
                    distort,
                    [&](const Distort &varDistort) {
                        auto varIntrinsics = this->intrinsics;
                        varIntrinsics.d.params = varDistort.data;
                        return varIntrinsics.computeDistortValueJacobian(normXY.data).value;
                    },
                    delta);
            CHECK_KMAT_CLOSE(distValJac.distortJac, distortRefDeriv, 1E-3f, seed);
        }
    }
};

#define RBA_DEFINE_IntrinsicsDerivativeTest(r, data, TRAITS)\
    using BOOST_PP_CAT(IntrinsicsDerivativeTest_, TRAITS) = IntrinsicsDerivativeTest<TRAITS>;\
    TEST_F(BOOST_PP_CAT(IntrinsicsDerivativeTest_, TRAITS), normXY)\
    {\
        testNormXY(0);\
    }\
    \
    TEST_F(BOOST_PP_CAT(IntrinsicsDerivativeTest_, TRAITS), distort)\
    {\
        testDistort(0);\
    }
BOOST_PP_SEQ_FOR_EACH(RBA_DEFINE_IntrinsicsDerivativeTest, data, ALL_TRAITS)
#undef RBA_DEFINE_IntrinsicsDerivativeTest