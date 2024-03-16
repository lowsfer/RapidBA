//
// Created by yao on 16/09/18.
//

#pragma once
#include <cstdint>
#include "../kmat.h"
#include "../utils_kernel.h"
#include "capture.h"
#include "intrinsic.h"
#include "../RapidBA.h"
#include "../DiscreteModel.h"
#include <type_traits>

namespace rba {

template <template <typename ftype, bool rolling> typename IntrinsicsTemplate, template <ShutterType> typename InterfaceTemplate, ShutterType shutterType>
struct TraitsGrpBase : FPTraitsSSD {
    using FPTraits = FPTraitsSSD;
	static constexpr ShutterType shutter = shutterType;
	static constexpr bool rolling = isRolling(shutter);
	static constexpr bool rollVGlb = isRollVGlb(shutterType);
    using CamIntr = IntrinsicsTemplate<typename FPTraits::lpf, rolling>;
    using Capture = GroupCaptureTemplate<shutter, typename FPTraits::lpf>;
	using Velocity = typename Capture::Velocity;
    using Intrinsics = CamIntr;
    using Interface = InterfaceTemplate<shutter>;
    using PubInterface = Interface;
	static_assert(shutter == Velocity::shutter);
};

#define INSTANTIATE_GRP_TRAITS(CFG) \
	using TraitsGrp##CFG##Global = TraitsGrp##CFG##Template<ShutterType::kGlobal>; \
	using TraitsGrp##CFG##Roll0D = TraitsGrp##CFG##Template<ShutterType::kRollingFixedVelocity>; \
	using TraitsGrp##CFG##Roll1D = TraitsGrp##CFG##Template<ShutterType::kRolling1D>; \
	using TraitsGrp##CFG##Roll1DLoc = TraitsGrp##CFG##Template<ShutterType::kRolling1DLoc>; \
	using TraitsGrp##CFG##Roll3D = TraitsGrp##CFG##Template<ShutterType::kRolling3D>;

template <ShutterType shutterType>
struct TraitsGrpF2C2D5Template : TraitsGrpBase<IntrinsicsF2C2D5, IGroupModelF2C2D5, shutterType> {
private:
	using Base = TraitsGrpBase<IntrinsicsF2C2D5, IGroupModelF2C2D5, shutterType>;
public:
    static typename Base::CamIntr toCamParams(const typename Base::Interface::CamParamType& src) {
        return {.f = {src.f.x, src.f.y}, .c = {src.c.x, src.c.y}, .d = {src.k1, src.k2, src.p1, src.p2, src.k3}, .rollingCenter = src.rollingCenter};
    }
    static typename Base::Interface::CamParamType toCamParams(const typename Base::CamIntr& src) {
        return {{src.f[0], src.f[1]}, {src.c[0], src.c[1]}, src.d.k1(), src.d.k2(), src.d.p1(), src.d.p2(), src.d.k3(), src.rollingCenter};
    }
};
INSTANTIATE_GRP_TRAITS(F2C2D5)

template <ShutterType shutterType>
struct TraitsGrpF2D5Template : TraitsGrpBase<IntrinsicsF2D5, IGroupModelF2D5, shutterType> {
private:
	using Base = TraitsGrpBase<IntrinsicsF2D5, IGroupModelF2D5, shutterType>;
public:
    static typename Base::CamIntr toCamParams(const typename Base::Interface::CamParamType& src) {
        return {.f = {src.f.x, src.f.y}, .d = {src.k1, src.k2, src.p1, src.p2, src.k3}, .rollingCenter = src.rollingCenter};
    }
    static typename Base::Interface::CamParamType toCamParams(const typename Base::CamIntr& src) {
        return {{src.f[0], src.f[1]}, src.d.k1(), src.d.k2(), src.d.p1(), src.d.p2(), src.d.k3(), src.rollingCenter};
    }
};
INSTANTIATE_GRP_TRAITS(F2D5)

template <ShutterType shutterType>
struct TraitsGrpF1C2D5Template : TraitsGrpBase<IntrinsicsF1C2D5, IGroupModelF1C2D5, shutterType>  {
private:
	using Base = TraitsGrpBase<IntrinsicsF1C2D5, IGroupModelF1C2D5, shutterType>;
public:
    static typename Base::CamIntr toCamParams(const typename Base::Interface::CamParamType& src) {
        return {src.f, {src.c.x, src.c.y}, {src.k1, src.k2, src.p1, src.p2, src.k3}, src.rollingCenter};
    }
    static typename Base::Interface::CamParamType toCamParams(const typename Base::CamIntr& src) {
        return {src.f, {src.c[0], src.c[1]}, src.d.k1(), src.d.k2(), src.d.p1(), src.d.p2(), src.d.k3(), src.rollingCenter};
    }
};
INSTANTIATE_GRP_TRAITS(F1C2D5)

template <ShutterType shutterType>
struct TraitsGrpF1D5Template : TraitsGrpBase<IntrinsicsF1D5, IGroupModelF1D5, shutterType>  {
private:
	using Base = TraitsGrpBase<IntrinsicsF1D5, IGroupModelF1D5, shutterType>;
public:
    static typename Base::CamIntr toCamParams(const typename Base::Interface::CamParamType& src) {
        return {src.f, {src.k1, src.k2, src.p1, src.p2, src.k3}, src.rollingCenter};
    }
    static typename Base::Interface::CamParamType toCamParams(const typename Base::CamIntr& src) {
        return {src.f, src.d.k1(), src.d.k2(), src.d.p1(), src.d.p2(), src.d.k3(), src.rollingCenter};
    }
};
INSTANTIATE_GRP_TRAITS(F1D5)

template <ShutterType shutterType>
struct TraitsGrpF1D2Template : TraitsGrpBase<IntrinsicsF1D2, IGroupModelF1D2, shutterType> {
private:
	using Base = TraitsGrpBase<IntrinsicsF1D2, IGroupModelF1D2, shutterType>;
public:
    static typename Base::CamIntr toCamParams(const typename Base::Interface::CamParamType& src) {
        return {src.f, {src.k1, src.k2}, src.rollingCenter};
    }
    static typename Base::Interface::CamParamType toCamParams(const typename Base::CamIntr& src) {
        return {src.f, src.d.k1(), src.d.k2(), src.rollingCenter};
    }
};
INSTANTIATE_GRP_TRAITS(F1D2)

template <ShutterType shutterType>
struct TraitsGrpF1Template : TraitsGrpBase<IntrinsicsF1, IGroupModelF1, shutterType> {
private:
	using Base = TraitsGrpBase<IntrinsicsF1, IGroupModelF1, shutterType>;
public:
    static typename Base::CamIntr toCamParams(const typename Base::Interface::CamParamType& src) {
        return {src.f, {}, src.rollingCenter};
    }
    static typename Base::Interface::CamParamType toCamParams(const typename Base::CamIntr& src) {
        return {src.f, src.rollingCenter};
    }
};
INSTANTIATE_GRP_TRAITS(F1)

template <ShutterType shutterType>
struct TraitsGrpF2Template : TraitsGrpBase<IntrinsicsF2, IGroupModelF2, shutterType> {
private:
	using Base = TraitsGrpBase<IntrinsicsF2, IGroupModelF2, shutterType>;
public:
    static typename Base::CamIntr toCamParams(const typename Base::Interface::CamParamType& src) {
        return {{src.f.x, src.f.y}, {}, src.rollingCenter};
    }
    static typename Base::Interface::CamParamType toCamParams(const typename Base::CamIntr& src) {
        return {{src.f[0], src.f[1]}, src.rollingCenter};
    }
};
INSTANTIATE_GRP_TRAITS(F2)

template <ShutterType shutterType>
struct TraitsGrpF2C2Template : TraitsGrpBase<IntrinsicsF2C2, IGroupModelF2C2, shutterType> {
private:
	using Base = TraitsGrpBase<IntrinsicsF2C2, IGroupModelF2C2, shutterType>;
public:
    static typename Base::CamIntr toCamParams(const typename Base::Interface::CamParamType& src) {
        return {{src.f.x, src.f.y}, {src.c.x, src.c.y}, {}, src.rollingCenter};
    }
    static typename Base::Interface::CamParamType toCamParams(const typename Base::CamIntr& src) {
        return {{src.f[0], src.f[1]}, {src.c[0], src.c[1]}, src.rollingCenter};
    }
};
INSTANTIATE_GRP_TRAITS(F2C2)
#undef INSTANTIATE_GRP_TRAITS

struct TraitsDisF1D2 : FPTraitsSSD {
    using FPTraits = FPTraitsSSD;
    using CamIntr = IntrinsicsNull<typename FPTraits::lpf>;
    using Capture = CaptureF1D2<typename FPTraits::lpf>;
	using Velocity = Capture::Velocity;
	static constexpr ShutterType shutter = Velocity::shutter;
	static constexpr bool rollVGlb = isRollVGlb(shutter);
    using Intrinsics = typename Capture::Intrinsics;
    using Interface = IDiscreteModelF1D2Impl;
    using PubInterface = IDiscreteModelF1D2;
    static CamIntr toCamParams(const typename Interface::CamParamType& /*src*/) {
        return CamIntr{};
    }
    static typename Interface::CamParamType toCamParams(const CamIntr& /*src*/) {
        return {};
    }
};

struct TraitsDisF1 : FPTraitsSSD {
    using FPTraits = FPTraitsSSD;
    using CamIntr = IntrinsicsNull<typename FPTraits::lpf>;
    using Capture = CaptureF1<typename FPTraits::lpf>;
	using Velocity = Capture::Velocity;
	static constexpr ShutterType shutter = Velocity::shutter;
	static constexpr bool rollVGlb = isRollVGlb(shutter);
    using Intrinsics = typename Capture::Intrinsics;
    using Interface = IDiscreteModelF1Impl;
    using PubInterface = IDiscreteModelF1;
    static CamIntr toCamParams(const typename Interface::CamParamType& /*src*/) {
        return CamIntr{};
    }
    static typename Interface::CamParamType toCamParams(const CamIntr& /*src*/) {
        return {};
    }
};

template <typename Traits>
constexpr bool isGroupModel(){
    constexpr bool isGrouped = std::is_same<typename Traits::CamIntr, typename Traits::Intrinsics>::value;
    constexpr bool isDiscrete = std::is_same<typename Traits::CamIntr, IntrinsicsNull<typename Traits::lpf>>::value;
    static_assert(isGrouped != isDiscrete, "fatal error");
    return isGrouped;
}

template <typename CameraType, typename CaptureType> __device__ __host__ __forceinline__
std::enable_if_t<CameraType::DoF != 0u, CameraType> getIntrinsics(const CameraType& camera, const CaptureType& capture)
{
    return camera;
}

template <typename CameraType, typename CaptureType> __device__ __host__ __forceinline__
std::enable_if_t<CameraType::DoF == 0u, typename CaptureType::Intrinsics> getIntrinsics(const CameraType& camera, const CaptureType& capture)
{
    return capture.intrinsics;
}

template <typename Traits> __device__ __host__ __forceinline__
typename Traits::Velocity getVelocity(const typename Traits::Capture& cap) {
	if constexpr (!isRolling(Traits::shutter)) {
		return VelocityNull<typename Traits::lpf>{};
	}
	else {
		return cap.velocity;
	}
#ifdef __CUDACC__
	return {}; // nvcc complains about missing return statement
#endif
}
template <typename Traits> __device__ __host__ __forceinline__
typename Traits::lpf getRollingCenter(const typename Traits::Intrinsics& intri) {
	if constexpr (!isRolling(Traits::Velocity::shutter)) {
		return NAN;
	}
	else {
		return intri.rollingCenter;
	}
#ifdef __CUDACC__
	return {}; // nvcc complains about missing return statement
#endif
}

} // namespace rba

#define ALL_TRAITS \
	(TraitsGrpF2C2D5Global) (TraitsGrpF2D5Global) (TraitsGrpF1C2D5Global) (TraitsGrpF1D5Global) (TraitsGrpF1D2Global) (TraitsGrpF1Global) (TraitsGrpF2Global) (TraitsGrpF2C2Global) \
	(TraitsGrpF2C2D5Roll0D) (TraitsGrpF2D5Roll0D) (TraitsGrpF1C2D5Roll0D) (TraitsGrpF1D5Roll0D) (TraitsGrpF1D2Roll0D) (TraitsGrpF1Roll0D) (TraitsGrpF2Roll0D) (TraitsGrpF2C2Roll0D) \
	(TraitsGrpF2C2D5Roll1D) (TraitsGrpF2D5Roll1D) (TraitsGrpF1C2D5Roll1D) (TraitsGrpF1D5Roll1D) (TraitsGrpF1D2Roll1D) (TraitsGrpF1Roll1D) (TraitsGrpF2Roll1D) (TraitsGrpF2C2Roll1D) \
	(TraitsGrpF2C2D5Roll1DLoc) (TraitsGrpF2D5Roll1DLoc) (TraitsGrpF1C2D5Roll1DLoc) (TraitsGrpF1D5Roll1DLoc) (TraitsGrpF1D2Roll1DLoc) (TraitsGrpF1Roll1DLoc) (TraitsGrpF2Roll1DLoc) (TraitsGrpF2C2Roll1DLoc) \
	(TraitsGrpF2C2D5Roll3D) (TraitsGrpF2D5Roll3D) (TraitsGrpF1C2D5Roll3D) (TraitsGrpF1D5Roll3D) (TraitsGrpF1D2Roll3D) (TraitsGrpF1Roll3D) (TraitsGrpF2Roll3D) (TraitsGrpF2C2Roll3D) \
	(TraitsDisF1D2) (TraitsDisF1)
#define ALL_DISCRETE_TRAITS (TraitsDisF1D2) (TraitsDisF1)

#define RBA_IMPORT_TRAITS(Traits) \
    using CamIntr = typename Traits::CamIntr;\
    using Capture = typename Traits::Capture;\
    using Intrinsics = typename Traits::Intrinsics;\
    RBA_IMPORT_FPTYPES(Traits)
