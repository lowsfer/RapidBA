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
// Created by yao on 14/09/18.
//
#pragma once

#include "kernel.h"
#include "kmat.h"
namespace rba {

//point in camera coordinate system and jacobians
template <typename ftype, bool rollVGlb>
struct TransPointDerivative
{
    kmat<ftype, 3> position;//point in camera coordinate system
    //jacobians
    // deltaGvec*R0*(X - (C0+deltaC)), derivative to deltaGvec and deltaC. cJac is always -ptJac;
    struct {
        kmat<ftype, 3, 3> gvec;
        kmat<ftype, 3, Point<ftype>::DoF> pt;
        constexpr kmat<ftype, 3, 3> c() const {return -pt;}
		constexpr kmat<ftype, 3, 3> rollingDelta() const {
			if constexpr (rollVGlb) {
				return c();
			}
			else {
				return kmat<ftype, 3, 3>::eye();
			}
		}
    } jacobian;
};

template <ShutterType shutter, typename ftype>
__device__ __host__ __forceinline__
TransPointDerivative<ftype, isRollVGlb(shutter)> computeTransPointDerivative(const kmat<ftype,3,3>& R, const kmat<Coordinate<ftype>,3>& C, const kmat<Coordinate<ftype>,3>& p0, const kmat<ftype, 3>& rollingDelta)
{
	constexpr bool rolling = isRolling(shutter);
	constexpr bool rollVGlb = isRollVGlb(shutter);
    TransPointDerivative<ftype, rollVGlb> result;
    const kmat<ftype, 3> p = p0 - C;
	if constexpr (rolling) {
		if constexpr (rollVGlb) {
			result.position = R * (p - rollingDelta);
		}
		else {
    		result.position = R * p + rollingDelta;
		}
	}
	else {
		assert(rollingDelta.sqrNorm() == 0);
		result.position = R * p;
	}
    result.jacobian.gvec = [&](){
        const ftype x0 = p[0] * 2, x1 = p[1] * 2, x2 = p[2] * 2;
        const ftype j01 = R(2,0)*x0 + R(2,1)*x1 + R(2,2)*x2;
        const ftype j20 = R(1,0)*x0 + R(1,1)*x1 + R(1,2)*x2;
        const ftype j12 = R(0,0)*x0 + R(0,1)*x1 + R(0,2)*x2;
        return kmat<ftype, 3, 3>({
            0, j01, -j20,
            -j01, 0, j12,
            j20, -j12, 0
        });
    }();
    result.jacobian.pt = R;
    return result;
}

template <typename Traits>
struct ErrorDerivative{
    RBA_IMPORT_TRAITS(Traits);
    using ftype = lpf;
    kmat<ftype, 2, 1> error;
    //jacobians
    struct Jacobian{
        kmat<ftype, 2u, CamIntr::DoF> camera; // rename to camera
        kmat<ftype, 2u, Capture::DoF> capture; // rename to capture
        kmat<ftype, 2, 3> pt;
        template <bool isGroup = isGroupModel<Traits>()> __device__ __host__ __forceinline__
        static std::enable_if_t<isGroup,Jacobian> make(const kmat<ftype, 2, Intrinsics::DoF>& intri,
            const kmat<ftype, 2u, Pose<ftype>::DoF + Traits::Velocity::DoF>& extri, const kmat<ftype, 2, 3>& pt)
        {
            static_assert(CamIntr::DoF == Intrinsics::DoF, "fatal error");
            return {intri, extri, pt};
        }
        template <bool isGroup = isGroupModel<Traits>()> __device__ __host__ __forceinline__
        static std::enable_if_t<!isGroup,Jacobian> make(const kmat<ftype, 2, Intrinsics::DoF>& intri,
            const kmat<ftype, 2u, Pose<ftype>::DoF>& extri, const kmat<ftype, 2, 3>& pt)
        {
            static_assert(Capture::DoF == Intrinsics::DoF + Pose<ftype>::DoF, "fatal error");
            kmat<ftype, 2u, Capture::DoF> cap;
            cap.assignBlock(0u, 0u, extri);
            cap.assignBlock(0u, extri.cols(), intri);
            return {{}, cap, pt};
        }
#ifndef __CUDACC__
        kmat<ftype, 2, Intrinsics::DoF> intrinsics() const {
            if constexpr (isGroupModel<Traits>()){
                return camera;
            }
            else{
                return capture.template block<2u, Intrinsics::DoF>(0u, Pose<ftype>::DoF);
            }
        }
        kmat<ftype, 2, Pose<ftype>::DoF> pose() const {
            return capture.template block<2u, Pose<ftype>::DoF>(0, 0);
        }
#endif
    };
    
    Jacobian jacobian;
    __device__ __host__ __forceinline__
    static ErrorDerivative zeros(){
        return {
            kmat<ftype, 2, 1>::zeros(),
            {
                kmat<ftype, 2, CamIntr::DoF>::zeros(),
                kmat<ftype, 2, Capture::DoF>::zeros(),
                kmat<ftype, 2, 3>::zeros()
            }
        };
    }
};

template <typename Traits>
__device__ __host__ __forceinline__
ErrorDerivative<Traits> computeErrorDerivativeImpl(const typename Traits::Intrinsics& intrinsics,
        const kmat<typename Traits::lpf,3,3>& capR, const kmat<typename Traits::locf, 3>& capC,
		const typename Traits::Velocity& velocity, typename Traits::lpf verticalCenter, // verticalCenter may be moved into intrinsics.
        const kmat<typename Traits::locf, 3>& p0, const kmat<typename Traits::lpf, 2>& observation)
{
    RBA_IMPORT_TRAITS(Traits);
    using ftype = lpf;

	kmat<lpf, 3> rollingDelta; // translation due to rolling shutter in T or C fashion, depending on Traits::rollVGlb
	const ftype time = observation[1] - verticalCenter;
	if constexpr(isRolling(Traits::shutter)) {
		rollingDelta = velocity.getVelocity() * time;
	}
	else {
		unused(time);
		rollingDelta = kmat<lpf, 3>::zeros();
	}
    const TransPointDerivative<ftype, Traits::rollVGlb> transPtDerivative = computeTransPointDerivative<Traits::shutter>(capR, capC, p0, rollingDelta);

    const typename Intrinsics::ValueJacobian valueDerivative = intrinsics.computeValueDerivative(transPtDerivative.position);

	kmat<ftype, 2, Pose<ftype>::DoF + Traits::Velocity::DoF> extriJac;
    kmat<ftype, 2, Pose<ftype>::DoF> transJac;
    transJac.assignBlock(0,0, valueDerivative.jacobian.pt * transPtDerivative.jacobian.gvec);
	const kmat<ftype, 2, Point<ftype>::DoF> valCJac = valueDerivative.jacobian.pt * transPtDerivative.jacobian.c(); // cJac = -ptJac in transPtDerivative
    transJac.assignBlock(0,3, valCJac);
	extriJac.assignBlock(0, 0, transJac);
	if constexpr(Traits::Velocity::DoF != 0) {
		kmat<ftype, 2, Traits::Velocity::DoF> velocityJac;
		if constexpr (Traits::shutter == ShutterType::kRolling1D) {
			velocityJac = valCJac * velocity.normal * time;
		}
		else if constexpr (Traits::shutter == ShutterType::kRolling1DLoc) {
			velocityJac = valueDerivative.jacobian.pt * /*valueDerivative.jacobian.rollingDelta() * */velocity.normal * time; // valueDerivative.jacobian.rollingDelta() is identity matrix
		}
		else if constexpr (Traits::shutter == ShutterType::kRolling3D) {
			velocityJac = valCJac * time;
		}
		else {
			static_assert(alwaysFalse<typename Traits::Velocity>());
		}
		extriJac.assignBlock(0u, Traits::Capture::vDofOffset, velocityJac);
	}
    return ErrorDerivative<Traits>{
        valueDerivative.value - observation, // Note that the SBA paper uses x-f(x), not f(x)-x. As a result, the computed delta needs negation
        ErrorDerivative<Traits>::Jacobian::make(
            valueDerivative.jacobian.intri,
            extriJac,
            valueDerivative.jacobian.pt * transPtDerivative.jacobian.pt
        )
    };
}

template <typename Traits>
__device__ __host__ __forceinline__
ErrorDerivative<Traits> updateForFixedCapLoc(const ErrorDerivative<Traits>& x, const kmat<bool, 3>& fixedCapLoc) {
    ErrorDerivative<Traits> y = x;
    using Capture = typename Traits::Capture;
    using Pose = decltype(std::declval<Capture>().pose);
    using lpf = typename Traits::lpf;
    constexpr uint32_t cDofOffset = Capture::poseDofOffset + Pose::cDofOffset;
    static_assert(cDofOffset + 3 <= Capture::DoF);
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        if (fixedCapLoc[i]) {
            y.jacobian.capture.assignBlock(0u, cDofOffset + i, kmat<lpf, 2>::zeros());
        }
    }
    return y;            
}

template <typename Traits>
__device__ __host__ __forceinline__
ErrorDerivative<Traits> computeErrorDerivative(const typename Traits::CamIntr& camera,
        const typename Traits::Capture& capture,
        const kmat<typename Traits::locf, 3>& p0, const kmat<typename Traits::lpf, 2>& observation,
        const kmat<bool, 3>& fixedCapLoc = {false, false, false})
{
    const auto intrinsics = getIntrinsics(camera, capture);
    const auto pose = capture.getPose();
    using lpf = typename Traits::lpf;
    ErrorDerivative<Traits> errDer = computeErrorDerivativeImpl<Traits>(intrinsics, quat2mat(kmat<lpf, 4>{pose.q}),
        kmat<typename Traits::locf, 3>{pose.c},
		getVelocity<Traits>(capture), getRollingCenter<Traits>(intrinsics),
		p0, observation);
    return updateForFixedCapLoc(errDer, fixedCapLoc);
}

template <typename ftype>
struct CtrlErrorDerivative
{
    ftype error;
    sym::One jacobian;
};
template <typename ftype>
__device__ __host__ __forceinline__
CtrlErrorDerivative<ftype> computeCtrlErrorDerivative(const Coordinate<ftype>& x, const Coordinate<ftype>& ctrlLoc)
{
    return CtrlErrorDerivative<ftype>{x - ctrlLoc, sym::One{}};
}

template <typename ftype> __device__ __host__ __forceinline__
kmat<ftype, 3> computeCtrlError(const kmat<Coordinate<ftype>, 3>& x, const kmat<Coordinate<ftype>, 3>& ctrlLoc) {
    kmat<ftype, 3> error;
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        const auto errDer = computeCtrlErrorDerivative(x[i], ctrlLoc[i]);
        assert(float(errDer.jacobian) == 1.f); // jacobian is identity and we ignore it
        error[i] = errDer.error;
    }
    return error;
}

//refer to g2o implementation, e.g. core/base_unary_edge.hpp
// robustChi2 contains (weighted) omega.
template<typename ftype, uint32_t D>
__device__ __host__ __forceinline__
ftype robustifyByHuber(kmat<ftype, D> error, ftype omega, ftype delta, ftype* robustChi2 = nullptr){
	assert(std::isfinite(omega));
    ftype weightedOmega;
    const ftype sqrErrOmega = error.sqrNorm() * omega;
    const auto sqrDelta = sqr(delta);
    constexpr float ceiling = 10;
    if (sqrErrOmega < sqrDelta){
        weightedOmega = omega;
        if (robustChi2)
            *robustChi2 = sqrErrOmega;
    }
    else if (sqrErrOmega < sqrDelta * sqr(ceiling)){
        weightedOmega = omega * delta * fast_rsqrt(sqrErrOmega);
        if (robustChi2)
            *robustChi2 = 2*fast_sqrt(sqrErrOmega)*delta - sqrDelta;
    }
    else {
        weightedOmega = 0.f;
        if (robustChi2)
            *robustChi2 = (2*ceiling - 1) * sqrDelta;
    }
	// may trigger after a wrong huge update of the model, when LM lambda is too small.
	assert(robustChi2 == nullptr || std::isfinite(*robustChi2));
    return weightedOmega;
}

template<typename ftype, uint32_t D>
__device__ __host__ __forceinline__
symkmat<ftype, D> robustifyByHuber(kmat<ftype, D> error, symkmat<ftype, D> omega, ftype delta, ftype* robustChi2 = nullptr){
	assert(omega.allFinite());
    symkmat<ftype, D> weightedOmega;
    const ftype sqrErrOmega = (error.transpose() * omega.toKMat() * error).scalar();
    const auto sqrDelta = sqr(delta);
    constexpr float ceiling = 10;
    if (sqrErrOmega < sqrDelta){
        weightedOmega = omega;
        if (robustChi2)
            *robustChi2 = sqrErrOmega;
    }
    else if (sqrErrOmega < sqrDelta * sqr(ceiling)){
        weightedOmega = symkmat{omega.toKMat() * (delta * fast_rsqrt(sqrErrOmega))};
        if (robustChi2)
            *robustChi2 = 2*fast_sqrt(sqrErrOmega)*delta - sqrDelta;
    }
    else {
        weightedOmega = symkmat<ftype, D>{};
        if (robustChi2)
            *robustChi2 = (2*ceiling - 1) * sqrDelta;
    }
	// may trigger after a wrong huge update of the model, when LM lambda is too small.
	assert(robustChi2 == nullptr || std::isfinite(*robustChi2));
    return weightedOmega;
}

// SC/DCS from "At All Costs: A Comparison of Robust Cost Functions for Camera Correspondence Outliers"
// It's different from DCS from other sources, and also different from G2O.
template<typename ftype, uint32_t D>
__device__ __host__ __forceinline__
ftype robustifyByDCS(kmat<ftype, D> error, ftype omega, ftype delta, ftype* robustChi2 = nullptr){
	assert(std::isfinite(omega));
    ftype weightedOmega;
    const ftype e2 = error.sqrNorm() * omega;
    const auto phi = sqr(delta);
    if (e2 < phi) {
        weightedOmega = omega;
        if (robustChi2)
            *robustChi2 = e2;
    }
    else {
        weightedOmega = omega * 4*sqr(phi)/sqr(e2 + phi);
        if (!std::isfinite(weightedOmega)) {
            weightedOmega = 0;
        }
        if (robustChi2){
            *robustChi2 = 4*phi*e2 / (phi+e2) - phi;
            if (!std::isfinite(*robustChi2)) {
                *robustChi2 = 3 * phi;
            }
        }
    }
	assert(robustChi2 == nullptr || std::isfinite(*robustChi2));
    return weightedOmega;
}
template<typename ftype, uint32_t D>
__device__ __host__ __forceinline__
symkmat<ftype, D> robustifyByDCS(kmat<ftype, D> error, symkmat<ftype, D> omega, ftype delta, ftype* robustChi2 = nullptr){
	assert(omega.allFinite());
    symkmat<ftype, D> weightedOmega;
    const ftype e2 = (error.transpose() * omega.toKMat() * error).scalar();
    const auto phi = sqr(delta);
    if (e2 < phi) {
        weightedOmega = omega;
        if (robustChi2)
            *robustChi2 = e2;
    }
    else {
        auto const rho1 =  4*sqr(phi)/sqr(e2 + phi);
        weightedOmega = symkmat{omega.toKMat() * rho1};
        if (!std::isfinite(rho1)) {
            weightedOmega = symkmat<ftype, D>{};
        }
        if (robustChi2){
            *robustChi2 = 4*phi*e2 / (phi+e2) - phi;
            if (!std::isfinite(*robustChi2)) {
                *robustChi2 = 3 * phi;
            }
        }
    }
	assert(robustChi2 == nullptr || std::isfinite(*robustChi2));
    return weightedOmega;
}

template<typename ftype, uint32_t D>
__device__ __host__ __forceinline__
ftype robustify(kmat<ftype, D> error, ftype omega, ftype delta, ftype* robustChi2 = nullptr){
    return robustifyByDCS<ftype, D>(error, omega, delta, robustChi2);
}
template<typename ftype, uint32_t D>
__device__ __host__ __forceinline__
symkmat<ftype, D> robustify(kmat<ftype, D> error, symkmat<ftype, D> omega, ftype delta, ftype* robustChi2 = nullptr){
    return robustifyByDCS<ftype, D>(error, omega, delta, robustChi2);
}
} // namespace rba

