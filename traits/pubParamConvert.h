#include "../RapidBA.h"
#include "intrinsic.h"
#include "capture.h"

namespace rba
{
template <typename ftype, bool rolling>
IModel::IntrinsicsTypeF2C2D5<rolling> convert(const IntrinsicsF2C2D5<ftype, rolling>& src){
    return IModel::IntrinsicsTypeF2C2D5{
        {src.fx(), src.fy()}, {src.cx(), src.cy()},
        src.d.k1(), src.d.k2(), src.d.p1(), src.d.p2(), src.d.k3(),
		src.rollingCenter
    };
}
template <typename ftype = float, bool rolling = false>
IntrinsicsF2C2D5<ftype, rolling> convert(const IModel::IntrinsicsTypeF2C2D5<rolling>& src){
    return IntrinsicsF2C2D5<ftype, rolling>{
        {src.f.x, src.f.y}, {src.c.x, src.c.y},
        {src.k1, src.k2, src.p1, src.p2, src.k3},
		src.rollingCenter
    };
}
template <typename ftype, bool rolling>
IModel::IntrinsicsTypeF2D5<rolling> convert(const IntrinsicsF2D5<ftype, rolling>& src){
    return IModel::IntrinsicsTypeF2D5{
        {src.fx(), src.fy()},
        src.d.k1(), src.d.k2(), src.d.p1(), src.d.p2(), src.d.k3(),
		src.rollingCenter
    };
}
template <typename ftype = float, bool rolling = false>
IntrinsicsF2D5<ftype, rolling> convert(const IModel::IntrinsicsTypeF2D5<rolling>& src){
    return IntrinsicsF2D5<ftype, rolling>{
        {src.f.x, src.f.y},
        {src.k1, src.k2, src.p1, src.p2, src.k3},
		src.rollingCenter
    };
}
template <typename ftype, bool rolling>
IModel::IntrinsicsTypeF1C2D5<rolling> convert(const IntrinsicsF1C2D5<ftype, rolling>& src){
    return IModel::IntrinsicsTypeF1C2D5<rolling>{
        src.f, {src.cx(), src.cy()},
        src.d.k1(), src.d.k2(), src.d.p1(), src.d.p2(), src.d.k3(),
		src.rollingCenter
    };
}
template <typename ftype = float, bool rolling = false>
IntrinsicsF1C2D5<ftype, rolling> convert(const IModel::IntrinsicsTypeF1C2D5<rolling>& src){
    return IntrinsicsF1C2D5<ftype, rolling>{
        src.f, {src.c.x, src.c.y},
        {src.k1, src.k2, src.p1, src.p2, src.k3},
		src.rollingCenter
    };
}
template <typename ftype, bool rolling>
IModel::IntrinsicsTypeF1D5<rolling> convert(const IntrinsicsF1D5<ftype, rolling>& src){
    return IModel::IntrinsicsTypeF1D5<rolling>{
        src.f,
        src.d.k1(), src.d.k2(), src.d.p1(), src.d.p2(), src.d.k3(),
		src.rollingCenter
    };
}
template <typename ftype = float, bool rolling = false>
IntrinsicsF1D5<ftype, rolling> convert(const IModel::IntrinsicsTypeF1D5<rolling>& src){
    return IntrinsicsF1D5<ftype, rolling>{
        src.f,
        {src.k1, src.k2, src.p1, src.p2, src.k3},
		src.rollingCenter
    };
}
template <typename ftype, bool rolling>
IModel::IntrinsicsTypeF1D2<rolling> convert(const IntrinsicsF1D2<ftype, rolling>& src){
    return IModel::IntrinsicsTypeF1D2<rolling>{
        src.f, src.d.k1(), src.d.k2(),
		.rollingCenter = src.rollingCenter
    };
}
template <typename ftype = float, bool rolling = false>
IntrinsicsF1D2<ftype, rolling> convert(const IModel::IntrinsicsTypeF1D2<rolling>& src){
    return IntrinsicsF1D2<ftype, rolling>{
        src.f, {src.k1, src.k2},
		.rollingCenter = src.rollingCenter
    };
}
template <typename ftype, bool rolling>
IModel::IntrinsicsTypeF1<rolling> convert(const IntrinsicsF1<ftype, rolling>& src){
    return IModel::IntrinsicsTypeF1<rolling>{src.f, .rollingCenter = src.rollingCenter};
}
template <typename ftype = float, bool rolling>
IntrinsicsF1<ftype, rolling> convert(const IModel::IntrinsicsTypeF1<rolling>& src){
    return IntrinsicsF1<ftype, rolling>{src.f, .rollingCenter = src.rollingCenter};
}
template <typename ftype>
grouped::CamParamTypeNull<false> convert(const IntrinsicsNull<ftype>& /*src*/){
    return {};
}
template <typename ftype = float>
IntrinsicsNull<ftype> convert(const grouped::CamParamTypeNull<false>& /*src*/){
    return {};
}

template <typename ftype>
IModel::Pose<false> convert(const Pose<ftype>& src) {
    return IModel::Pose<false>{
        {.x=src.q[1], .y=src.q[2], .z=src.q[3], .w=src.q[0]},
        {src.c[0].value, src.c[1].value, src.c[2].value}
    };
}
template <typename ftype = float>
Pose<ftype> convert(const IModel::Pose<false>& src) {
    auto toCoord = [](double src) -> Coordinate<ftype>{
        return {static_cast<typename Coordinate<ftype>::ImplType>(src)};
    };
    return Pose<ftype>{
        {src.q.w, src.q.x, src.q.y, src.q.z},
        {toCoord(src.c.x), toCoord(src.c.y), toCoord(src.c.z)}
    };
}

discrete::CapParamTypeF1D2<false> convert(const CaptureF1D2<float>& capture){
    return {convert(capture.pose), convert(capture.intrinsics)};
}

discrete::CapParamTypeF1<false> convert(const CaptureF1<float>& capture){
    return {convert(capture.pose), convert(capture.intrinsics)};
}

template <ShutterType shutter, typename ftype>
grouped::CapParamTypePose<isRolling(shutter)> convert(const GroupCaptureTemplate<shutter, ftype>& capture){
    const IModel::Pose<false> p = convert(capture.pose);
	if constexpr(isRolling(shutter)) {
		const auto v = capture.getVelocity();
		return {p.q, p.c, {v[0], v[1], v[2]}};
	}
	else {
		return p;
	}
}

template <typename KernelCaptureType, typename InterfaceCaptureType>
KernelCaptureType makeCapParams(const InterfaceCaptureType& capture, IdxCam idxCam);
namespace {
template <ShutterType shutter, typename ftype>
VelocityType<shutter, ftype> makeVelocity(const std::conditional_t<isRolling(shutter), float3, std::monostate>& v) {
	if constexpr (isRolling(shutter)) {
		return VelocityType<shutter, ftype>::make({v.x, v.y, v.z});
	}
	return {};
}
template <ShutterType shutter, typename ftype> 
GroupCaptureTemplate<shutter, ftype> makeGrpCapParams(const grouped::CapParamTypePose<isRolling(shutter)>& capture, IdxCam idxCam){
	IModel::Pose<false> p{capture.q, capture.c, {}};
	return {idxCam, convert(p), makeVelocity<shutter, ftype>(capture.velocity)};
}
}
template <>
GroupCaptureGlobal<float> makeCapParams<GroupCaptureGlobal<float>, grouped::CapParamTypePose<false>>(const grouped::CapParamTypePose<false>& capture, IdxCam idxCam){
    return makeGrpCapParams<ShutterType::kGlobal, float>(capture, idxCam);
}
template <>
GroupCaptureRoll0D<float> makeCapParams<GroupCaptureRoll0D<float>, grouped::CapParamTypePose<true>>(const grouped::CapParamTypePose<true>& capture, IdxCam idxCam){
    return makeGrpCapParams<ShutterType::kRollingFixedVelocity, float>(capture, idxCam);
}
template <>
GroupCaptureRoll1D<float> makeCapParams<GroupCaptureRoll1D<float>, grouped::CapParamTypePose<true>>(const grouped::CapParamTypePose<true>& capture, IdxCam idxCam){
    return makeGrpCapParams<ShutterType::kRolling1D, float>(capture, idxCam);
}
template <>
GroupCaptureRoll1DLoc<float> makeCapParams<GroupCaptureRoll1DLoc<float>, grouped::CapParamTypePose<true>>(const grouped::CapParamTypePose<true>& capture, IdxCam idxCam){
    return makeGrpCapParams<ShutterType::kRolling1DLoc, float>(capture, idxCam);
}
template <>
GroupCaptureRoll3D<float> makeCapParams<GroupCaptureRoll3D<float>, grouped::CapParamTypePose<true>>(const grouped::CapParamTypePose<true>& capture, IdxCam idxCam){
    return makeGrpCapParams<ShutterType::kRolling3D, float>(capture, idxCam);
}

template <>
CaptureF1D2<float> makeCapParams<CaptureF1D2<float>, discrete::CapParamTypeF1D2<false>>(const discrete::CapParamTypeF1D2<false>& capture, IdxCam idxCam){
    assert(idxCam == CaptureF1D2<float>::intrIdx);
    return {convert(capture.pose), convert(capture.intrinsics)};
}
template <>
CaptureF1<float> makeCapParams<CaptureF1<float>, discrete::CapParamTypeF1<false>>(const discrete::CapParamTypeF1<false>& capture, IdxCam idxCam){
    assert(idxCam == CaptureF1<float>::intrIdx);
    return {convert(capture.pose), convert(capture.intrinsics)};
}

} // namespace rba
