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
// Created by yao on 28/10/18.
//

#include "TestModel.h"
#include "../derivative.h"
#include "utils_test.h"
#include "../GroupModelTypes.h"
#include <boost/preprocessor/seq/for_each.hpp>

template <typename Traits, typename Func>
void perturbGrpModel(rba::grouped::HostModel<Traits>& hostModel, float ptsRange, const Func& genRand);

template <typename Traits, typename Func>
void makeRandomGrpModel(rba::grouped::HostModel<Traits>& hostModel, size_t nbPts, size_t nbIntri, size_t nbCaps,
                        float omega, float huber, float ptsRange, size_t minObsPerCap, const Func& genRand) {
    RBA_IMPORT_TRAITS(Traits);
    using namespace rba;
    // set up points
    assert(hostModel.pointList.empty());
    hostModel.pointList.resize(nbPts);
    std::generate(hostModel.pointList.begin(), hostModel.pointList.end(), [&](){return rba::Point<lpf>{{genRand()*ptsRange, genRand()*ptsRange, genRand()*ptsRange}};});
    // set up intrinsics
    assert(hostModel.intriList.empty());
    if constexpr(isGroupModel<Traits>()) {
        hostModel.intriList.resize(nbIntri);
        for (auto &intri : hostModel.intriList) {
            intri = RandParamGen<typename Traits::Intrinsics>::make(genRand);
        }
    }
    else{
        assert(nbIntri == 0);
    }
    // set up extrinsics
    const Eigen::Matrix<lpf, 3, 1> transRefPt = (Eigen::Matrix<lpf, 3, 1>{} << 0.f,0.f,1.5f*ptsRange).finished();
    hostModel.capList.resize(nbCaps);
    for (unsigned n = 0; n < nbCaps; n++){
        auto &cap = hostModel.capList[n];
        do {
            // set up rotation
            typename Eigen::Matrix<lpf, 4, 1>::MapType qMap(&cap.capture.pose.q[0]);
            const Eigen::Matrix<lpf, 3, 1> axis = Eigen::Matrix<lpf, 3, 1>::NullaryExpr(3, genRand).normalized();
            const lpf angle = genRand() * float(M_PI * 2);
            qMap << std::cos(angle / 2), std::sin(angle / 2) * axis;
            // set up translation
            const auto &refPt = hostModel.pointList[unsigned(genRand() * nbPts * 2 + nbPts) % nbPts];
            const Eigen::Quaternion<lpf> q(qMap[0], qMap[1], qMap[2], qMap[3]);
            Eigen::Matrix<typename locf::ImplType, 3, 1>::Map(&cap.capture.pose.c[0].value) =
                    Eigen::Matrix<typename locf::ImplType, 3, 1>::Map(&refPt.position[0].value) - (q.conjugate() * transRefPt).template cast<typename locf::ImplType>();
            // set up idxIntri
            if constexpr(isGroupModel<Traits>()) {
                cap.capture.intrIdx = n / divUp(unsigned(nbCaps), unsigned(nbIntri));
                assert(cap.capture.intrIdx < nbIntri);
            }
            else{
                static_assert(decltype(cap.capture)::intrIdx == std::numeric_limits<uint32_t>::max(), "fatal error");
                cap.capture.intrinsics = RandParamGen<typename Traits::Intrinsics>::make(genRand);
            }
			// set up rolling shutter
			if constexpr (isRolling(Traits::shutter)) {
				const auto v = kmat<float, 3>{{genRand(), genRand(), genRand()}} * 1E-4f;
				// const auto v = kmat<float, 3>::zeros();
				cap.capture.velocity = Traits::Velocity::make(v);
				//@fixme: observations also needs to be updated if v is not zero
			}
            // set up observations
            const auto intri = getIntrinsics(
                    isGroupModel<Traits>() ? hostModel.intriList[cap.capture.intrIdx] : typename Traits::CamIntr{},
                    cap.capture);
            cap.obs.clear();
            for (unsigned idxPt = 0; idxPt < nbPts; idxPt++) {
                const auto &pt = hostModel.pointList[idxPt];
				kmat<lpf, 3> v = kmat<lpf, 3>::zeros();
				if constexpr (isRolling(Traits::shutter)) {
					v = getVelocity<Traits>(cap.capture).getVelocity();
				}
				const float rollingCenter = getRollingCenter<Traits>(intri);
				lpf y = rollingCenter;
				kmat<lpf, 2> proj;
				bool outOfFoV = false;
				while (true) {
					const bool isGlobal = (v.sqrNorm() == 0);
					const bool rollVGlb = Traits::rollVGlb;
					kmat<locf, 3> C = kmat<locf, 3>{cap.capture.pose.c};
					if (!isGlobal && rollVGlb) {
						C = C + v * (y - rollingCenter);
					}
					Eigen::Matrix<lpf, 3, 1> transPt = q * toEigenMap(kmat<locf, 3>{pt.position} - C);
					if (!isGlobal && !rollVGlb) {
						transPt += toEigenMap(v * (y - rollingCenter));
					}
					proj = intri.computeValueDerivative(transPt.data()).value;
					if ( transPt.norm() < 1E-1f
						|| std::abs(proj[0] - intri.cx()) > intri.fx()
						|| std::abs(proj[1] - intri.cy()) > intri.fy())
					{
						outOfFoV = true;
						break;
					}
					if (isGlobal || std::abs(proj[1] - y) < 1E-3f) {
						break;
					}
					y = proj[1];
				}
				if (!outOfFoV) {
					cap.obs.emplace_back(CapOb<lpf>{{proj[0], proj[1]}, idxPt, __float2half(omega), __float2half(huber)});
				}
            }
        } while(cap.obs.size() < minObsPerCap);
    }

    hostModel.pointVarMask = std::vector<bool>(hostModel.pointList.size(), true);
    if (hostModel.pointList.size() > 8)
        hostModel.pointVarMask[0] = false;
    hostModel.nbVarPoints = IdxPt(std::count(hostModel.pointVarMask.begin(), hostModel.pointVarMask.end(), true));
    hostModel.capVarMask = std::vector<bool>(hostModel.capList.size(), true);
    if (hostModel.capList.size() > 3)
        hostModel.capVarMask[0] = false;
    hostModel.nbVarCaps = IdxCap(std::count(hostModel.capVarMask.begin(), hostModel.capVarMask.end(), true));
    hostModel.intriVarMask = std::vector<bool>(hostModel.intriList.size(), true);
    if (hostModel.intriList.size() > 3)
        hostModel.intriVarMask[0] = false;
    hostModel.nbVarIntri = IdxCam(std::count(hostModel.intriVarMask.begin(), hostModel.intriVarMask.end(), true));
    hostModel.nbTotalObs = IdxOb(std::accumulate(hostModel.capList.begin(), hostModel.capList.end(), size_t(0), [](size_t acc, const typename rba::grouped::HostModel<Traits>::CaptureViews& a){return acc + a.obs.size();}));
}

template <typename Traits, typename Func>
void perturbGrpModel(rba::grouped::HostModel<Traits>& hostModel, float ptsRange, const Func& genRand)
{
	RBA_IMPORT_TRAITS(Traits);
    using namespace rba;
	for (unsigned i = 0; i < hostModel.pointList.size(); i++) {
		if (hostModel.pointVarMask[i]){
			lpf delta[rba::Point<lpf>::DoF];
			std::generate_n(delta, rba::Point<lpf>::DoF, [&](){return genRand() * ptsRange * 1E-2f;});
			hostModel.pointList[i].update(delta);
		}
	}
	//@fixme: fix for discrete models
	for (unsigned i = 0; i < hostModel.capList.size(); i++) {
		if (hostModel.capVarMask[i]){
			lpf delta[Capture::DoF];
			std::generate_n(delta, Capture::DoF, [&](){return genRand() * 1E-2f;});
			if constexpr (isRolling(Traits::shutter)) {
				std::generate_n(&delta[Capture::vDofOffset], Capture::Velocity::DoF, [&]{return genRand() * 1E-5f;});
			}
			hostModel.capList[i].capture.update(delta);
		}
	}
	if constexpr (isGroupModel<Traits>()) {
		for (unsigned i = 0; i < hostModel.intriList.size(); i++) {
			if (hostModel.intriVarMask[i]) {
				lpf delta[CamIntr::DoF];
				std::generate_n(delta, CamIntr::nbFCParams, [&]() { return genRand() * 10; });
				std::generate_n(&delta[CamIntr::nbFCParams], CamIntr::DoF - CamIntr::nbFCParams, [&]() { return genRand() * 1E-3f; });
			}
		}
	}
}

template <typename Traits, template <typename> class BaseHostModel>
void TestModelBase<Traits, BaseHostModel>::SetUp() {
    setUpModel(nbPts, nbIntri, nbCaps);
}

template <typename Traits, template <typename> class BaseHostModel>
void TestModelBase<Traits, BaseHostModel>::setUpModel(size_t nbPts_, size_t nbIntri_, size_t nbCaps_) {
    nbPts = nbPts_;
    nbIntri = nbIntri_;
    nbCaps = nbCaps_;
    const float ptsRange = 1.f;
    const size_t minObsPerCap = std::min(std::min(std::max(nbPts / 4, size_t(16)), nbPts), size_t(128));

	auto genRand = [this](){return dist(rng);};
    makeRandomGrpModel(*this, nbPts, nbIntri, nbCaps, omega, huber, ptsRange, minObsPerCap, genRand);
	refPointList = hostModel.pointList;
	refIntriList = hostModel.intriList;
	std::transform(hostModel.capList.begin(), hostModel.capList.end(), std::back_inserter(refCapList), [](const auto& x){return x.capture;});
	perturbGrpModel(hostModel, ptsRange, genRand);
}

template <typename Traits, template <typename> class BaseHostModel>
void TestModelBase<Traits, BaseHostModel>::computeHostJacobian() {
    RBA_IMPORT_TRAITS(Traits);
    const size_t nbObs = std::accumulate(hostModel.capList.begin(), hostModel.capList.end(), size_t(0), [](size_t acc, const typename rba::grouped::HostModel<Traits>::CaptureViews& cap){return acc + cap.obs.size();});
    Eigen::Matrix<lpf, -1, -1> fullJacobian = Eigen::Matrix<lpf, -1, -1>::Zero(nbObs*2, CamIntr::DoF * hostModel.intriList.size() + Capture::DoF * hostModel.capList.size() + rba::Point<lpf>::DoF * hostModel.pointList.size());
    hostError.resize(nbObs*2);
    uint32_t obIdx = 0;
    for (rba::IdxCap idxCap = 0; idxCap < hostModel.capList.size(); idxCap++){
        const auto& cap = hostModel.capList[idxCap];
        const auto intri = rba::isGroupModel<Traits>() ? hostModel.intriList[cap.capture.intrIdx] : CamIntr{};
        for (const auto& ob : cap.obs){
            const auto errDeriv = rba::computeErrorDerivative<Traits>(intri, cap.capture, {hostModel.pointList[ob.ptIdx].position}, {ob.position});
            hostError.template block<2, 1>(obIdx*2, 0) << errDeriv.error[0], errDeriv.error[1];
            auto jacobRows = fullJacobian.block(obIdx*2, 0, 2, fullJacobian.cols());
            unsigned offset = 0;
            jacobRows.template block<2, CamIntr::DoF>(0, offset + CamIntr::DoF * cap.capture.intrIdx)
                    = toEigenMap(errDeriv.jacobian.camera);
            offset += CamIntr::DoF * hostModel.intriList.size();
            jacobRows.template block<2, Capture::DoF>(0, offset + Capture::DoF * idxCap)
                    = toEigenMap(errDeriv.jacobian.capture);
            offset += Capture::DoF * hostModel.capList.size();
            jacobRows.template block<2, rba::Point<lpf>::DoF>(0, offset + rba::Point<lpf>::DoF * ob.ptIdx)
                    = toEigenMap(errDeriv.jacobian.pt);
            obIdx++;
        }
    }

    hostJacobian = Eigen::Matrix<lpf, -1, -1>::Zero(nbObs*2, CamIntr::DoF * hostModel.nbVarIntri + Capture::DoF * hostModel.nbVarCaps + rba::Point<lpf>::DoF * hostModel.nbVarPoints);
    uint32_t dstOffset = 0;
    uint32_t srcOffset = 0;
    const auto rows = fullJacobian.rows();
    for (rba::IdxCam idxIntri = 0; idxIntri < hostModel.intriList.size(); idxIntri++){
        constexpr auto cols = CamIntr::DoF;
        if(hostModel.intriVarMask[idxIntri]) {
            hostJacobian.block(0, dstOffset, rows, cols) = fullJacobian.block(0, srcOffset, rows, cols);
            dstOffset += cols;
        }
        srcOffset += cols;
    }
    for (rba::IdxCap i = 0; i < hostModel.capList.size(); i++){
        const auto cols = Capture::DoF;
        if (hostModel.capVarMask[i]){
            hostJacobian.block(0, dstOffset, rows, cols) = fullJacobian.block(0, srcOffset, rows, cols);
            dstOffset += cols;
        }
        srcOffset += cols;
    }
    for (rba::IdxPt i = 0; i < hostModel.pointList.size(); i++){
        const auto cols = rba::Point<lpf>::DoF;
        if ( hostModel.pointVarMask[i]){
            hostJacobian.block(0, dstOffset, rows, cols) = fullJacobian.block(0, srcOffset, rows, cols);
            dstOffset += cols;
        }
        srcOffset += cols;
    }
    assert(srcOffset == CamIntr::DoF * hostModel.intriList.size() + Capture::DoF * hostModel.capList.size() + rba::Point<lpf>::DoF * hostModel.pointList.size() && srcOffset == fullJacobian.cols());
    assert(dstOffset == hostJacobian.cols());
}

template <typename Traits, template <typename> class BaseHostModel>
void TestModelBase<Traits, BaseHostModel>::computeHostHessian() {
    const auto J = hostJacobian.template cast<epf>().eval();
    //@fixme: huber is not yet used
    hostHessian = J.transpose() * J * omega;
    hostHessianEpsilon = J.transpose() * hostError.template cast<epf>() * omega;
}

template <typename Traits, template <typename> class BaseHostModel>
void TestModelBase<Traits, BaseHostModel>::computeHostSchur() {
    const size_t dims[2] = {
            CamIntr::DoF * hostModel.nbVarIntri + Capture::DoF * hostModel.nbVarCaps,
            rba::Point<lpf>::DoF * hostModel.nbVarPoints};
    const auto MQU = hostHessian.block(0, 0, dims[0], dims[0]);
    const auto V = hostHessian.block(dims[0], dims[0], dims[1], dims[1]);
    const auto SW = hostHessian.block(0, dims[0], dims[0], dims[1]);
    hostSchur = MQU - SW * V.ldlt().solve(SW.transpose().eval());
    const auto EcEa = hostHessianEpsilon.block(0, 0, dims[0], 1);
    const auto Eb = hostHessianEpsilon.block(dims[0], 0, dims[1], 1);
    hostSchurEpsilon = EcEa - SW * V.ldlt().solve(Eb);
}

#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template class TestModelBase<rba::TRAITS, rba::grouped::HostModel>;\
    template class TestModelBase<rba::TRAITS, rba::grouped::GroupModel>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template <typename Traits>
void TestModel<Traits>::SetUp() {
    ::testing::Test::SetUp();
    TestModelBase<Traits, rba::grouped::GroupModel>::SetUp();
    devModel->init(*this);
    devInvVBlocks = deviceAlloc<VSymBlock>(this->nbVarPoints);
    devHessian->init(*this, devModel->involvedCaps.data(), devModel->involvedCaps.size(), devHessianPreComp.get(), true);
    devSchur->init(*devModel, *devHessian, devSchurPreComp.get(), nullptr);
}

#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template class TestModel<rba::TRAITS>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

//using Traits = rba::TraitsGrpF2C2D5Global;