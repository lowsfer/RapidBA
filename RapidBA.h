#ifndef RAPIDBA_LIBRARY_H
#define RAPIDBA_LIBRARY_H
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <array>
#include <variant>

namespace rba {
using IdxPt = uint32_t;
using IdxCap = uint32_t;
using IdxCam = uint32_t;

enum class ShutterType{
	kGlobal,
	kRollingFixedVelocity, // User specifies the velocity for each capture. Do not optimize velocity.
	kRolling1D,
	kRolling1DLoc, // velocity is in the local coordinate system. similar to T representation of pose, i.e. velocity of scene in the camera coordinate system
	kRolling3D
};

constexpr bool isRolling(ShutterType shutterType) {
	return shutterType != ShutterType::kGlobal;
}

constexpr bool isRollVGlb(ShutterType shutterType) {
	return shutterType != ShutterType::kRolling1DLoc;
}

class IModel {
public:
	template <bool rolling>
	struct Pose {
		float4 q;
		double3 c;
		std::conditional_t<rolling, float3, std::monostate> velocity; // In T fashion of pose representation for kRolling1DLoc and C fashion for the rest.
		static constexpr bool isRollingShutter = rolling;
	};
    enum class IntriType{
        kNull, // invalid
        kF2C2D5,
        kF2D5,
        kF2C2,
        kF2,
        kF1C2D5,
        kF1D5,
        kF1D2,
        kF1
    };
	template <bool rolling>
    struct IntrinsicsTypeF2C2D5{
        float2 f;
        float2 c;
        float k1;
        float k2;
        float p1;
        float p2;
        float k3;
        static constexpr IntriType intriType = IntriType::kF2C2D5;
		std::conditional_t<rolling, float, std::monostate> rollingCenter;
    };
	template <bool rolling>
    struct IntrinsicsTypeF2D5{
        float2 f;
        float k1;
        float k2;
        float p1;
        float p2;
        float k3;
        static constexpr IntriType intriType = IntriType::kF2D5;
		std::conditional_t<rolling, float, std::monostate> rollingCenter;
    };
	template <bool rolling>
    struct IntrinsicsTypeF2C2{
        float2 f;
        float2 c;
        static constexpr IntriType intriType = IntriType::kF2C2;
		std::conditional_t<rolling, float, std::monostate> rollingCenter;
    };
	template <bool rolling>
    struct IntrinsicsTypeF2{
        float2 f;
        static constexpr IntriType intriType = IntriType::kF2;
		std::conditional_t<rolling, float, std::monostate> rollingCenter;
    };
	template <bool rolling>
    struct IntrinsicsTypeF1C2D5{
        float f;
        float2 c;
        float k1;
        float k2;
        float p1;
        float p2;
        float k3;
        static constexpr IntriType intriType = IntriType::kF1C2D5;
		std::conditional_t<rolling, float, std::monostate> rollingCenter;
    };
	template <bool rolling>
    struct IntrinsicsTypeF1D5{
        float f;
        float k1;
        float k2;
        float p1;
        float p2;
        float k3;
        static constexpr IntriType intriType = IntriType::kF1D5;
		std::conditional_t<rolling, float, std::monostate> rollingCenter;
    };
	template <bool rolling>
    struct IntrinsicsTypeF1D2{
        float f;
        float k1;
        float k2;
        static constexpr IntriType intriType = IntriType::kF1D2;
		std::conditional_t<rolling, float, std::monostate> rollingCenter;
    };
	template <bool rolling>
    struct IntrinsicsTypeF1{
        float f;
        static constexpr IntriType intriType = IntriType::kF1;
		std::conditional_t<rolling, float, std::monostate> rollingCenter;
    };
    virtual IdxPt addPoint(double3 position, bool fixed = false) = 0;
    virtual double3 getPointPosition(IdxPt idx) const = 0;

    virtual void setPointFixed(IdxPt idx, bool fixed) = 0;
    virtual void setCaptureFixed(IdxCap idx, bool fixed) = 0;

    // @fixme: implemented but not tested
    // omega is the Fisher information matrix, i.e. the inversion of covariance matrix of position.
    // omega must be symmetric and positive definite
    // Use omega = all INFINITY if you want the capture location to be fixed in that dimension
    // We don't allow some dimensions to be hard while other dimensions to be soft.
    // omega elements should be all finite or all INFINITY
    virtual void setCaptureGNSS(IdxCap idx, double3 position, float omega[3][3], float huber = INFINITY) = 0;
    // @fixme: implemented but not tested
    // omega is the Fisher information matrix, i.e. the inversion of covariance matrix of position.
    // omega must be symmetric and positive definite
    // All omage elements must be finite. For hard control point, use setPointFixed() instead.
    // We don't allow some dimensions to be hard while other dimensions to be soft.
	// soft control points means the point 3d coordinate is allowed to change, the position is used as an extra constraint.
    virtual void setSoftCtrlPoint(IdxPt idx, double3 position, float omega[3][3], float huber = INFINITY) = 0;

    // filterModel() fixes numeric issues. Examples:
    //  1. For each variable camera, remove observation that is too close to image plane (i.e. transPt.z is small);
    //  2. every variable point should have at least two observing cameras that are not too far away.
    virtual void filterModel() = 0;

    virtual void initializeOptimization() = 0;
    virtual void setInitDamp(float damp) = 0; // Default: 1E-3f. Must init solver first
    // enum class OptStatus
    // {
    //     kSuccess,
    //     kNotConverged,
    //     kNumericError
    // };
    virtual void optimize(size_t maxIters/* = 64*/) = 0;

    virtual void clear() = 0; // @fixme: this is not working
    virtual void setVerbose(bool verbose) = 0;
    virtual IntriType getIntriType() const = 0;
    virtual bool isGrouped() const = 0;
	virtual ShutterType getShutterType() const;
	virtual bool hasRollingShutter() const;

    virtual ~IModel() = default;
};

// Monocular cameras, contrary to camera group with sync'ed exposure
class IMonoModel : public IModel
{
public:
    virtual void addObservation(IdxCap idxCap, IdxPt idxPt, float2 proj, float omega/* = 1.f*/, float huber/* = INFINITY*/) = 0;
};

/**************************************************************/
// Each image has its own intrinsics
template <typename CaptureParamsType>
class IDiscreteModel : public IMonoModel {
public:
    using CapParamType = CaptureParamsType;
    virtual IdxCap addCapture(const CapParamType& params, bool fixed = false) = 0;
    virtual void setCaptureParams(IdxCap idx, const CapParamType& params) = 0;
    virtual CapParamType getCaptureParams(IdxCap idx) const = 0;
    IntriType getIntriType() const final;
    bool isGrouped() const override {return false;}
};

namespace discrete
{
template <bool rolling = false>
struct CapParamTypeF1D2{
    typename IModel::Pose<rolling> pose;
    typename IModel::IntrinsicsTypeF1D2<rolling> intrinsics;
};
template <bool rolling = false>
struct CapParamTypeF1{
    typename IModel::Pose<rolling> pose;
    typename IModel::IntrinsicsTypeF1<rolling> intrinsics;
};
template <bool rolling = false>
using CapParamType0 = typename IModel::Pose<rolling>;
} // namespace discrete

using IDiscreteModelF1D2 = IDiscreteModel<discrete::CapParamTypeF1D2<false>>;
IDiscreteModelF1D2* createDiscreteModelF1D2();

using IDiscreteModelF1 = IDiscreteModel<discrete::CapParamTypeF1<false>>;
IDiscreteModelF1* createDiscreteModelF1();

using IDiscreteModel0 = IDiscreteModel<discrete::CapParamType0<false>>;
IDiscreteModel0* createDiscreteModel0();

/**************************************************************/
// Multiple images may share one set of intrinsics.
class IGroupModelBase : public IMonoModel
{
public:
    virtual void setCameraFixed(IdxCam idx, bool fixed) = 0;
    virtual IdxCam getIdxCamForCapture(IdxCap idxCap) const = 0;
    bool isGrouped() const override;
};
template <template<bool> class CameraParamsType, template<bool> class CaptureParamsType, ShutterType shutter>
class IGroupModel : public IGroupModelBase{
public:
	static constexpr ShutterType shutterType = shutter;
	static constexpr bool isRollingShutter = isRolling(shutter);
    using CamParamType = CameraParamsType<isRollingShutter>;
    using CapParamType = CaptureParamsType<isRollingShutter>;

    virtual IdxCap addCapture(IdxCam idxCam, const CapParamType& params, bool fixed = false) = 0;
    virtual void setCaptureParams(IdxCap idx, const CapParamType& params) = 0;
    virtual CapParamType getCaptureParams(IdxCap idx) const = 0;


    virtual IdxCam addCamera(const CamParamType& params, bool fixed = false) = 0;
    virtual CamParamType getCameraParams(IdxCam idx) const = 0;

	ShutterType getShutterType() const override {return shutterType;}
	bool hasRollingShutter() const override {return isRollingShutter;}
};

namespace grouped{
template <bool rolling> using CamParamTypeF2C2D5 = IModel::IntrinsicsTypeF2C2D5<rolling>;
template <bool rolling> using CamParamTypeF2D5 = IModel::IntrinsicsTypeF2D5<rolling>;
template <bool rolling> using CamParamTypeF1C2D5 = IModel::IntrinsicsTypeF1C2D5<rolling>;
template <bool rolling> using CamParamTypeF1D5 = IModel::IntrinsicsTypeF1D5<rolling>;
template <bool rolling> using CamParamTypeF2C2 = IModel::IntrinsicsTypeF2C2<rolling>;
template <bool rolling> using CamParamTypeF2 = IModel::IntrinsicsTypeF2<rolling>;
template <bool rolling> using CamParamTypeF1D2 = IModel::IntrinsicsTypeF1D2<rolling>;
template <bool rolling> using CamParamTypeF1 = IModel::IntrinsicsTypeF1<rolling>;

template <bool rolling> using CapParamTypePose = IModel::Pose<rolling>;
} // namespace grouped

template <ShutterType shutterType>
using IGroupModelF2C2D5 = IGroupModel<grouped::CamParamTypeF2C2D5, IModel::Pose, shutterType>;
template <ShutterType shutterType>
IGroupModelF2C2D5<shutterType>* createGroupModelF2C2D5();

template <ShutterType shutterType>
using IGroupModelF2D5 = IGroupModel<grouped::CamParamTypeF2D5, IModel::Pose, shutterType>;
template <ShutterType shutterType>
IGroupModelF2D5<shutterType>* createGroupModelF2D5();

template <ShutterType shutterType>
using IGroupModelF1C2D5 = IGroupModel<grouped::CamParamTypeF1C2D5, IModel::Pose, shutterType>;
template <ShutterType shutterType>
IGroupModelF1C2D5<shutterType>* createGroupModelF1C2D5();

template <ShutterType shutterType>
using IGroupModelF1D5 = IGroupModel<grouped::CamParamTypeF1D5, IModel::Pose, shutterType>;
template <ShutterType shutterType>
IGroupModelF1D5<shutterType>* createGroupModelF1D5();

template <ShutterType shutterType>
using IGroupModelF2C2 = IGroupModel<grouped::CamParamTypeF2C2, IModel::Pose, shutterType>;
template <ShutterType shutterType>
IGroupModelF2C2<shutterType>* createGroupModelF2C2();

template <ShutterType shutterType>
using IGroupModelF2 = IGroupModel<grouped::CamParamTypeF2, IModel::Pose, shutterType>;
template <ShutterType shutterType>
IGroupModelF2<shutterType>* createGroupModelF2();

template <ShutterType shutterType>
using IGroupModelF1D2 = IGroupModel<grouped::CamParamTypeF1D2, IModel::Pose, shutterType>;
template <ShutterType shutterType>
IGroupModelF1D2<shutterType>* createGroupModelF1D2();

template <ShutterType shutterType>
using IGroupModelF1 = IGroupModel<grouped::CamParamTypeF1, IModel::Pose, shutterType>;
template <ShutterType shutterType>
IGroupModelF1<shutterType>* createGroupModelF1();

using UniversalIntrinsics = grouped::CamParamTypeF2C2D5<true>;
class IUniversalModel : public IGroupModel<grouped::CamParamTypeF2C2D5, IModel::Pose, ShutterType::kRolling3D> {};

// Note that even if you set useGroupModel=false, it may still use group model internally if there is no discrete model for your intriType
IUniversalModel* createUniversalModel(bool useGroupModel, IModel::IntriType intriType, ShutterType shutterType = ShutterType::kGlobal);

/**************************************************************/
// Multiple images may share one set of intrinsics. And a group of images may share one set of pose.
template <typename IntrinsicsType>
class ISyncObliqueModel : public IModel {
public:
    static constexpr size_t maxLensPerCam = 9;
    struct CamGrpParamType{
        struct LensParamType{
            IntrinsicsType intrinsics;
            Pose<false> offset; // relative to capture pose. LensPose = offset*CapturePose
        };
        LensParamType lenses[maxLensPerCam];
        size_t nbLenses;
    };
    using CapParamType = Pose<false>;
    using IdxCamGrp = uint32_t;
    using IdxLens = uint32_t; // range: [0, nbLenses)

    virtual IdxCamGrp addCameraGroup(const CamGrpParamType& params, bool fixed = false) = 0;
    virtual void setCameraGroupFixed(IdxCam idx, bool fixed) = 0;
    virtual CamGrpParamType getCameraGroupParams(IdxCamGrp idx) const = 0;

    virtual IdxCap addCapture(IdxCamGrp idxCamGrp, const CapParamType& params, bool fixed = false) = 0;
    virtual CapParamType getCaptureParams(IdxCap idx) const = 0;

    virtual void addObservation(IdxCap idxCap, IdxPt idxPt, IdxLens idxLens, float2 proj, float omega/* = 1.f*/, float huber/* = INFINITY*/) = 0;
};

} // namespace rba

#endif // RAPIDBA_LIBRARY_H
