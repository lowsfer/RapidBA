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
// Created by yao on 2/23/20.
//

#include "UniversalModel.h"
#include "utils_host.h"
#include "utils_general.h"

namespace {
template <typename Dst, typename Src>
Dst semiDynCast(Src* src){
    static_assert(std::is_pointer<Dst>::value, "fatal error");
#ifndef NDEBUG
    const auto dst = dynamic_cast<Dst>(src);
    assert(dst != nullptr);
    return dst;
#else
    return static_cast<Dst>(src);
#endif
}
}

namespace rba {
using IntriType = IModel::IntriType;
template <ShutterType shutter>
std::unique_ptr<IGroupModelBase> createGrpModel(IntriType intriType) {
	switch (intriType) {
		case IntriType::kF1:
			return std::unique_ptr<IGroupModelBase>{createGroupModelF1<shutter>()};
		case IntriType::kF1D2:
			return std::unique_ptr<IGroupModelBase>{createGroupModelF1D2<shutter>()};
		case IntriType::kF1C2D5:
			return std::unique_ptr<IGroupModelBase>{createGroupModelF1C2D5<shutter>()};
        case IntriType::kF1D5:
            return std::unique_ptr<IGroupModelBase>{createGroupModelF1D5<shutter>()};
		case IntriType::kF2:
			return std::unique_ptr<IGroupModelBase>{createGroupModelF2<shutter>()};
		case IntriType::kF2C2:
			return std::unique_ptr<IGroupModelBase>{createGroupModelF2C2<shutter>()};
		case IntriType::kF2C2D5:
			return std::unique_ptr<IGroupModelBase>{createGroupModelF2C2D5<shutter>()};
		case IntriType::kF2D5:
			return std::unique_ptr<IGroupModelBase>{createGroupModelF2D5<shutter>()};
		case IntriType::kNull:
			throw std::runtime_error("Unsupported camera model");
	}
	throw std::runtime_error("You should never reach here");
}
std::unique_ptr<IMonoModel> createBundleModel(bool mayShareIntrinsics, IntriType intriType, ShutterType shutter) {
    if (!mayShareIntrinsics) {
		require(shutter == ShutterType::kGlobal);
        switch (intriType) {
            case IntriType::kF1:
                return std::unique_ptr<IMonoModel>{createDiscreteModelF1()};
            case IntriType::kF1D2:
                return std::unique_ptr<IMonoModel>{createDiscreteModelF1D2()};
            case IntriType::kF1C2D5:
            case IntriType::kF1D5:
            case IntriType::kF2:
            case IntriType::kF2C2:
            case IntriType::kF2C2D5:
            case IntriType::kF2D5:
            case IntriType::kNull:
                throw std::runtime_error("Unsupported camera model");
        }
    }
	else {
		switch (shutter) {
			case ShutterType::kGlobal: return createGrpModel<ShutterType::kGlobal>(intriType);
			case ShutterType::kRollingFixedVelocity: return createGrpModel<ShutterType::kRollingFixedVelocity>(intriType);
			case ShutterType::kRolling1D: return createGrpModel<ShutterType::kRolling1D>(intriType);
			case ShutterType::kRolling1DLoc: return createGrpModel<ShutterType::kRolling1DLoc>(intriType);
			case ShutterType::kRolling3D: return createGrpModel<ShutterType::kRolling3D>(intriType);
		}
    }
    throw std::runtime_error("You should never reach here");
}

UniversalModel::UniversalModel(bool useGroupModel, IModel::IntriType intriType, ShutterType shutter)
        : mIntriType{intriType} {
    if (useGroupModel) {
        mIsGrouped = true;
    } else {
        switch (intriType) {
            case IntriType::kF1:
            case IntriType::kF1D2:
                mIsGrouped = false;
                break;
            case IntriType::kF1C2D5:
            case IntriType::kF1D5:
            case IntriType::kF2:
            case IntriType::kF2C2:
            case IntriType::kF2C2D5:
            case IntriType::kF2D5:
                mIsGrouped = true;
                break;
            case IntriType::kNull:
                throw std::runtime_error("Unsupported camera model");
        }
    }
    mModel = createBundleModel(mIsGrouped, mIntriType, shutter);
}

UniversalModel::~UniversalModel() = default;

void UniversalModel::setVerbose(bool verbose) {
    mModel->setVerbose(verbose);
}

IdxPt UniversalModel::addPoint(double3 position, bool fixed) {
    return mModel->addPoint(position, fixed);
}

void UniversalModel::setCaptureFixed(IdxCap idx, bool fixed) {
    mModel->setCaptureFixed(idx, fixed);
}

void UniversalModel::setPointFixed(IdxPt idx, bool fixed) {
    mModel->setPointFixed(idx, fixed);
}

void UniversalModel::setCaptureGNSS(IdxCap idx, double3 position, float omega[3][3], float huber) {
    mModel->setCaptureGNSS(idx, position, omega, huber);
}

void UniversalModel::setSoftCtrlPoint(IdxPt idx, double3 position, float omega[3][3], float huber) {
    mModel->setSoftCtrlPoint(idx, position, omega, huber);
}

double3 UniversalModel::getPointPosition(IdxPt idx) const {
    return mModel->getPointPosition(idx);
}

void UniversalModel::filterModel() {
    mModel->filterModel();
}

void UniversalModel::clear() {
    mModel->clear();
	mCamCenters.clear();
}

void UniversalModel::setInitDamp(float damp) {
    mModel->setInitDamp(damp);
}

void UniversalModel::initializeOptimization() {
    mModel->initializeOptimization();
}

void UniversalModel::optimize(size_t maxIters) {
    mModel->optimize(maxIters);
}

IModel::IntriType UniversalModel::getIntriType() const {
    require(mIntriType == mModel->getIntriType());
    return mIntriType;
}

namespace {
inline void requireZero(float x) {require(x == 0);}
inline void requireZero(float2 x) {require(x.x == 0 && x.y == 0);}
template <typename T, typename... Args>
inline void requireZero(T v, const Args&... args) {
    requireZero(v);
    requireZero(args...);
}
template <bool rolling>
inline auto rollingCenterFromUniversal(float src) {
	if constexpr (rolling) {
		return src;
	}
	else {
		return std::monostate{};
	}
}
template <typename Intrinsics>
Intrinsics fromUniversal(const UniversalIntrinsics& src);
#define DEFINE_fromUniversal_SPECIALIZATION(rolling) \
	template <> \
	IModel::IntrinsicsTypeF2C2D5<rolling> fromUniversal<IModel::IntrinsicsTypeF2C2D5<rolling>>(const UniversalIntrinsics& s){ \
		return {s.f, s.c, s.k1, s.k2, s.p1, s.p2, s.k3, rollingCenterFromUniversal<rolling>(s.rollingCenter)}; \
	} \
	template <> \
	IModel::IntrinsicsTypeF2D5<rolling> fromUniversal<IModel::IntrinsicsTypeF2D5<rolling>>(const UniversalIntrinsics& s){ \
		return {s.f, s.k1, s.k2, s.p1, s.p2, s.k3, rollingCenterFromUniversal<rolling>(s.rollingCenter)}; \
	} \
	template <> \
	IModel::IntrinsicsTypeF2C2<rolling> fromUniversal<IModel::IntrinsicsTypeF2C2<rolling>>(const UniversalIntrinsics& s){ \
		requireZero(s.k1, s.k2, s.p1, s.p2, s.k3); \
		return {s.f, s.c, rollingCenterFromUniversal<rolling>(s.rollingCenter)}; \
	} \
	template <> \
	IModel::IntrinsicsTypeF2<rolling> fromUniversal<IModel::IntrinsicsTypeF2<rolling>>(const UniversalIntrinsics& s) { \
		requireZero(s.k1, s.k2, s.p1, s.p2, s.k3); \
		return {s.f, rollingCenterFromUniversal<rolling>(s.rollingCenter)}; \
	} \
	template <> \
	IModel::IntrinsicsTypeF1<rolling> fromUniversal<IModel::IntrinsicsTypeF1<rolling>>(const UniversalIntrinsics& s) { \
		requireZero(s.k1, s.k2, s.p1, s.p2, s.k3); \
		require(s.f.x == s.f.y); \
		return {s.f.x, rollingCenterFromUniversal<rolling>(s.rollingCenter)}; \
	} \
	template <> \
	IModel::IntrinsicsTypeF1D2<rolling> fromUniversal<IModel::IntrinsicsTypeF1D2<rolling>>(const UniversalIntrinsics& s){ \
		require(s.f.x == s.f.y); \
		requireZero(s.p1, s.p2, s.k3); \
		return {s.f.x, s.k1, s.k2, rollingCenterFromUniversal<rolling>(s.rollingCenter)}; \
	} \
	template <> \
	IModel::IntrinsicsTypeF1C2D5<rolling> fromUniversal<IModel::IntrinsicsTypeF1C2D5<rolling>>(const UniversalIntrinsics& s){ \
		require(s.f.x == s.f.y); \
		return {s.f.x, s.c, s.k1, s.k2, s.p1, s.p2, s.k3, rollingCenterFromUniversal<rolling>(s.rollingCenter)}; \
	} \
	template <> \
	IModel::IntrinsicsTypeF1D5<rolling> fromUniversal<IModel::IntrinsicsTypeF1D5<rolling>>(const UniversalIntrinsics& s){ \
		require(s.f.x == s.f.y); \
		return {s.f.x, s.k1, s.k2, s.p1, s.p2, s.k3, rollingCenterFromUniversal<rolling>(s.rollingCenter)}; \
	}
DEFINE_fromUniversal_SPECIALIZATION(true)
DEFINE_fromUniversal_SPECIALIZATION(false)
#undef DEFINE_fromUniversal_SPECIALIZATION
template <bool rolling>
IModel::Pose<rolling> fromUniversal(const IModel::Pose<true>& s) {
	if constexpr (rolling) {
		return {s.q, s.c, s.velocity};
	}
	else {
		require((std::isnan(s.velocity.x) && std::isnan(s.velocity.y) && std::isnan(s.velocity.z)) || (s.velocity.x == 0 || s.velocity.y == 0 || s.velocity.z == 0));
		return {s.q, s.c, std::monostate{}};
	}	
}

template <bool rolling>
inline float rollingCenterToUniversal(std::conditional_t<rolling, float, std::monostate> src) {
	if constexpr (rolling) {
		return src;
	}
	else {
		return NAN;
	}
}
template <typename Intrinsics>
UniversalIntrinsics toUniversal(const Intrinsics& i);
#define DEFINE_toUniversal_SPECIALIZATION(rolling) \
	template <> \
	UniversalIntrinsics toUniversal<IModel::IntrinsicsTypeF2C2D5<rolling>>(const IModel::IntrinsicsTypeF2C2D5<rolling>& s){ \
		return {s.f, s.c, s.k1, s.k2, s.p1, s.p2, s.k3, rollingCenterToUniversal<rolling>(s.rollingCenter)}; \
	} \
	template <> \
	UniversalIntrinsics toUniversal<IModel::IntrinsicsTypeF2D5<rolling>>(const IModel::IntrinsicsTypeF2D5<rolling>& s){ \
		return {s.f, {0,0}, s.k1, s.k2, s.p1, s.p2, s.k3, rollingCenterToUniversal<rolling>(s.rollingCenter)}; \
	} \
	template <> \
	UniversalIntrinsics toUniversal<IModel::IntrinsicsTypeF2C2<rolling>>(const IModel::IntrinsicsTypeF2C2<rolling>& s){ \
		return {s.f, s.c, 0, 0, 0, 0, 0, rollingCenterToUniversal<rolling>(s.rollingCenter)}; \
	} \
	template <> \
	UniversalIntrinsics toUniversal<IModel::IntrinsicsTypeF2<rolling>>(const IModel::IntrinsicsTypeF2<rolling>& s) { \
		return {s.f, {0,0}, 0, 0, 0, 0, 0, rollingCenterToUniversal<rolling>(s.rollingCenter)}; \
	} \
	template <> \
	UniversalIntrinsics toUniversal<IModel::IntrinsicsTypeF1<rolling>>(const IModel::IntrinsicsTypeF1<rolling>& s) { \
		return {{s.f, s.f}, {0,0}, 0, 0, 0, 0, 0, rollingCenterToUniversal<rolling>(s.rollingCenter)}; \
	} \
	template <> \
	UniversalIntrinsics toUniversal<IModel::IntrinsicsTypeF1D2<rolling>>(const IModel::IntrinsicsTypeF1D2<rolling>& s){ \
		return {{s.f, s.f}, {0,0}, s.k1, s.k2, 0, 0, 0, rollingCenterToUniversal<rolling>(s.rollingCenter)}; \
	} \
	template <> \
	UniversalIntrinsics toUniversal<IModel::IntrinsicsTypeF1C2D5<rolling>>(const IModel::IntrinsicsTypeF1C2D5<rolling>& s){ \
		return {{s.f, s.f}, s.c, s.k1, s.k2, s.p1, s.p2, s.k3, rollingCenterToUniversal<rolling>(s.rollingCenter)}; \
	} \
	template <> \
	UniversalIntrinsics toUniversal<IModel::IntrinsicsTypeF1D5<rolling>>(const IModel::IntrinsicsTypeF1D5<rolling>& s){ \
		return {{s.f, s.f}, {0,0}, s.k1, s.k2, s.p1, s.p2, s.k3, rollingCenterToUniversal<rolling>(s.rollingCenter)}; \
	}
DEFINE_toUniversal_SPECIALIZATION(true)
DEFINE_toUniversal_SPECIALIZATION(false)
#undef DEFINE_toUniversal_SPECIALIZATION
template <bool rolling>
IModel::Pose<true> toUniversal(const IModel::Pose<rolling>& s) {
	if constexpr (rolling) {
		return IModel::Pose<true>{s.q, s.c, s.velocity};
	}
	else {
		return IModel::Pose<true>{s.q, s.c, {0.f, 0.f, 0.f}};
	}
}

}

IdxCam UniversalModel::addCamera(const UniversalIntrinsics &params, bool fixed) {
	switch (mModel->getShutterType()) {
#define CASE(x) case ShutterType::x: return addCameraImpl<ShutterType::x>(params, fixed)
		CASE(kGlobal);
		CASE(kRollingFixedVelocity);
		CASE(kRolling1D);
		CASE(kRolling1DLoc);
		CASE(kRolling3D);
#undef CASE
	}
	fail();
}
template <ShutterType shutter>
IdxCam UniversalModel::addCameraImpl(const UniversalIntrinsics& params, bool fixed) {
	constexpr bool rolling = (isRolling(shutter));
    IModel* const model = mModel.get();
	require(shutter == model->getShutterType());
	require(model->hasRollingShutter() == rolling);
    const Pose<rolling> invalidPose {{NAN, NAN, NAN, NAN}, {NAN, NAN, NAN}}; unused(invalidPose);
    const float2 c = params.c;
    const auto centerlessParams = [&params](){
        auto x = params;
        x.c = {0.f, 0.f};
        return x;
    }();
    switch(getIntriType()){
        case IntriType::kNull:
            throw std::runtime_error("Invalid intrinsics type");
        case IntriType::kF2C2D5:
            assert(isGrouped());
            return semiDynCast<IGroupModelF2C2D5<shutter>*>(model)->addCamera(fromUniversal<IntrinsicsTypeF2C2D5<rolling>>(params), fixed);
        case IntriType::kF2D5: {
            assert(isGrouped());
            const auto idxCam =
                semiDynCast<IGroupModelF2D5<shutter>*>(model)->addCamera(fromUniversal<IntrinsicsTypeF2D5<rolling>>(centerlessParams), fixed);
            require(mCamCenters.try_emplace(idxCam, c).second);
            return idxCam;
        }
        case IntriType::kF2C2:
            assert(isGrouped());
            return semiDynCast<IGroupModelF2C2<shutter>*>(model)->addCamera(fromUniversal<IntrinsicsTypeF2C2<rolling>>(params), fixed);
        case IntriType::kF2:{
            assert(isGrouped());
            const auto idxCam =
                semiDynCast<IGroupModelF2<shutter>*>(model)->addCamera(fromUniversal<IntrinsicsTypeF2<rolling>>(centerlessParams), fixed);
            require(mCamCenters.try_emplace(idxCam, c).second || !isGrouped());
            return idxCam;
        }
        case IntriType::kF1C2D5:
            assert(isGrouped());
            return semiDynCast<IGroupModelF1C2D5<shutter>*>(model)->addCamera(fromUniversal<IntrinsicsTypeF1C2D5<rolling>>(params), fixed);
        case IntriType::kF1D5: {
            assert(isGrouped());
            const auto idxCam =
                semiDynCast<IGroupModelF1D5<shutter>*>(model)->addCamera(fromUniversal<IntrinsicsTypeF1D5<rolling>>(centerlessParams), fixed);
            require(mCamCenters.try_emplace(idxCam, c).second);
            return idxCam;
        }
        case IntriType::kF1D2: {
            IdxCam idxCam = badIdx<IdxCam>();
            if (isGrouped()) {
                idxCam = semiDynCast<IGroupModelF1D2<shutter>*>(model)->addCamera(
                        fromUniversal<IntrinsicsTypeF1D2<rolling>>(centerlessParams), fixed);
            } else {
				if constexpr (!rolling) {
					const IdxCap idxCap = semiDynCast<IDiscreteModelF1D2 *>(model)->addCapture(
							{invalidPose, fromUniversal<IntrinsicsTypeF1D2<rolling>>(params)}, fixed);
					idxCam = static_cast<IdxCam>(idxCap);
				}
				else {
					require(!"discrete models does not support rolling shutter");
				}
            }
            require(mCamCenters.try_emplace(idxCam, c).second || !isGrouped());
            return idxCam;
        }
        case IntriType::kF1: {
            IdxCam idxCam = badIdx<IdxCam>();
            if (isGrouped()) {
                idxCam = semiDynCast<IGroupModelF1<shutter>*>(model)->addCamera(fromUniversal<IntrinsicsTypeF1<rolling>>(params), fixed);
            } else {
				if constexpr (!rolling) {
					const IdxCap idxCap = semiDynCast<IDiscreteModelF1 *>(model)->addCapture(
							{invalidPose, fromUniversal<IntrinsicsTypeF1<rolling>>(params)}, fixed);
					idxCam = static_cast<IdxCam>(idxCap);
				}
				else {
					require(!"discrete models does not support rolling shutter");
				}
            }
            require(mCamCenters.try_emplace(idxCam, c).second || !isGrouped());
            return idxCam;
        }
    }
    fail();
}

UniversalIntrinsics UniversalModel::getCameraParams(IdxCam idx) const {
	switch (mModel->getShutterType()) {
#define CASE(x) case ShutterType::x: return getCameraParamsImpl<ShutterType::x>(idx)
		CASE(kGlobal);
		CASE(kRollingFixedVelocity);
		CASE(kRolling1D);
		CASE(kRolling1DLoc);
		CASE(kRolling3D);
#undef CASE
	}
	fail();
}

template <ShutterType shutter>
UniversalIntrinsics UniversalModel::getCameraParamsImpl(IdxCam idx) const {
	constexpr bool rolling = isRolling(shutter);
    IModel* const model = mModel.get();
	require(model->getShutterType() == shutter);
	require(model->hasRollingShutter() == rolling);
    switch (getIntriType()){
        case IntriType::kNull:
            throw std::runtime_error("Invalid IntriType");
        case IntriType::kF2C2D5:
            assert(isGrouped());
            return toUniversal(semiDynCast<IGroupModelF2C2D5<shutter>*>(model)->getCameraParams(idx));
        case IntriType::kF2D5: {
            assert(isGrouped());
            auto params = toUniversal(semiDynCast<IGroupModelF2D5<shutter>*>(model)->getCameraParams(idx));
            requireZero(params.c);
            params.c = mCamCenters.at(idx);
            return params;
        }
        case IntriType::kF2C2:
            assert(isGrouped());
            return toUniversal(semiDynCast<IGroupModelF2C2<shutter>*>(model)->getCameraParams(idx));
        case IntriType::kF2: {
            assert(isGrouped());
            auto params = toUniversal(semiDynCast<IGroupModelF2<shutter>*>(model)->getCameraParams(idx));
            requireZero(params.c);
            params.c = mCamCenters.at(idx);
            return params;
        }
        case IntriType::kF1C2D5:
            assert(isGrouped());
            return toUniversal(semiDynCast<IGroupModelF1C2D5<shutter>*>(model)->getCameraParams(idx));
        case IntriType::kF1D5: {
            assert(isGrouped());
            auto params = toUniversal(semiDynCast<IGroupModelF1D5<shutter>*>(model)->getCameraParams(idx));
            requireZero(params.c);
            params.c = mCamCenters.at(idx);
            return params;
        }
        case IntriType::kF1D2: {
            UniversalIntrinsics params;
            if (isGrouped()) {
                params = toUniversal(semiDynCast<IGroupModelF1D2<shutter>*>(model)->getCameraParams(idx));
            } else {
				if constexpr (!rolling) {
                	params = toUniversal(semiDynCast<IDiscreteModelF1D2 *>(model)->getCaptureParams(idx).intrinsics);
				}
				else {
					fail();
				}
            }
            requireZero(params.c);
            params.c = mCamCenters.at(idx);
            return params;
        }
        case IntriType::kF1: {
            UniversalIntrinsics params;
            if (isGrouped()) {
                params = toUniversal(semiDynCast<IGroupModelF1<shutter> *>(model)->getCameraParams(idx));
            } else {
				if constexpr (!rolling) {
                	params = toUniversal(semiDynCast<IDiscreteModelF1 *>(model)->getCaptureParams(idx).intrinsics);
				}
				else {
					fail();
				}
            }
            requireZero(params.c);
            params.c = mCamCenters.at(idx);
            return params;
        }
    }
    fail();
}

IdxCap UniversalModel::addCapture(IdxCam idxCam, const IModel::Pose<true> &params, bool fixed) {
	switch (mModel->getShutterType()) {
#define CASE(x) case ShutterType::x: return addCaptureImpl<ShutterType::x>(idxCam, params, fixed)
		CASE(kGlobal);
		CASE(kRollingFixedVelocity);
		CASE(kRolling1D);
		CASE(kRolling1DLoc);
		CASE(kRolling3D);
#undef CASE
	}
	fail();
}

template <ShutterType shutter>
IdxCap UniversalModel::addCaptureImpl(IdxCam idxCam, const IModel::Pose<true> &params, bool fixed) {
	constexpr bool rolling = isRolling(shutter);
    IModel* const model = mModel.get();
	require(model->getShutterType() == shutter);
	require(model->hasRollingShutter() == rolling);
    switch (getIntriType()){
        case IntriType::kNull:
            throw std::runtime_error("invalid IntriType");
        case IntriType::kF2C2D5:
            return semiDynCast<IGroupModelF2C2D5<shutter>*>(model)->addCapture(idxCam, fromUniversal<rolling>(params), fixed);
        case IntriType::kF2D5:
            return semiDynCast<IGroupModelF2D5<shutter>*>(model)->addCapture(idxCam, fromUniversal<rolling>(params), fixed);
        case IntriType::kF2C2:
            return semiDynCast<IGroupModelF2C2<shutter>*>(model)->addCapture(idxCam, fromUniversal<rolling>(params), fixed);
        case IntriType::kF2:
            return semiDynCast<IGroupModelF2<shutter>*>(model)->addCapture(idxCam, fromUniversal<rolling>(params), fixed);
        case IntriType::kF1C2D5:
            return semiDynCast<IGroupModelF1C2D5<shutter>*>(model)->addCapture(idxCam, fromUniversal<rolling>(params), fixed);
        case IntriType::kF1D5:
            return semiDynCast<IGroupModelF1D5<shutter>*>(model)->addCapture(idxCam, fromUniversal<rolling>(params), fixed);
        case IntriType::kF1D2:
            if (isGrouped()) {
                return semiDynCast<IGroupModelF1D2<shutter>*>(model)->addCapture(idxCam, fromUniversal<rolling>(params), fixed);
            }
            else {
				if constexpr(!rolling) {
					const auto m = semiDynCast<IDiscreteModelF1D2*>(model);
					const IdxCap idxCap = static_cast<IdxCap>(idxCam);
					const auto intrinsics = m->getCaptureParams(idxCap).intrinsics;
					assert(std::isnan(m->getCaptureParams(idxCap).pose.q.w));
					m->setCaptureParams(idxCap, {fromUniversal<rolling>(params), intrinsics});
					model->setCaptureFixed(idxCap, fixed);
					return idxCap;
				}
				else {
					fail();
				}
            }
        case IntriType::kF1:
            if (isGrouped()){
                return semiDynCast<IGroupModelF1<shutter>*>(model)->addCapture(idxCam, fromUniversal<rolling>(params), fixed);
            }
            else {
				if constexpr (!rolling) {
					const auto m = semiDynCast<IDiscreteModelF1*>(model);
					const IdxCap idxCap = static_cast<IdxCap>(idxCam);
					const auto intrinsics = m->getCaptureParams(idxCap).intrinsics;
					assert(std::isnan(m->getCaptureParams(idxCap).pose.q.w));
					m->setCaptureParams(idxCap, {fromUniversal<rolling>(params), intrinsics});
					model->setCaptureFixed(idxCap, fixed);
					return idxCap;
				}
				else {
					fail();
				}
            }
    }
    throw std::runtime_error("fatal error");
}

void UniversalModel::addObservation(IdxCap idxCap, IdxPt idxPt, float2 proj, float omega, float huber) {
    switch (getIntriType()) {
        case IntriType::kNull:
            throw std::runtime_error("invalid IntriType");
        case IntriType::kF2C2D5:
        case IntriType::kF2C2:
        case IntriType::kF1C2D5:
            break;
        case IntriType::kF2:
        case IntriType::kF1D2:
        case IntriType::kF1:
        case IntriType::kF1D5:
        case IntriType::kF2D5: {
            const IdxCam idxCam = getIdxCamForCapture(idxCap);
            const float2 c = mCamCenters.at(idxCam);
            proj.x -= c.x;
            proj.y -= c.y;
            break;
        }
    }
    mModel->addObservation(idxCap, idxPt, proj, omega, huber);
}

void UniversalModel::setCameraFixed(IdxCam idx, bool fixed) {
    if (isGrouped()) {
        semiDynCast<IGroupModelBase*>(mModel.get())->setCameraFixed(idx, fixed);
    }
    else {
        throw std::runtime_error("Camera fixture is bundled with pose fixture in this mode");
    }
}

IModel::Pose<true> UniversalModel::getCaptureParams(IdxCap idx) const {
	switch (mModel->getShutterType()) {
#define CASE(x) case ShutterType::x: return getCaptureParamsImpl<ShutterType::x>(idx)
		CASE(kGlobal);
		CASE(kRollingFixedVelocity);
		CASE(kRolling1D);
		CASE(kRolling1DLoc);
		CASE(kRolling3D);
#undef CASE
	}
	fail();
}

template <ShutterType shutter>
IModel::Pose<true> UniversalModel::getCaptureParamsImpl(IdxCap idx) const {
	constexpr bool rolling = isRolling(shutter);
    IModel* const model = mModel.get();
	require(model->getShutterType() == shutter);
	require(model->hasRollingShutter() == rolling);
    switch (getIntriType()){
        case IntriType::kNull:
            throw std::runtime_error("invalid IntriType");
        case IntriType::kF2C2D5:
            return toUniversal(semiDynCast<IGroupModelF2C2D5<shutter>*>(model)->getCaptureParams(idx));
        case IntriType::kF2D5:
            return toUniversal(semiDynCast<IGroupModelF2D5<shutter>*>(model)->getCaptureParams(idx));
        case IntriType::kF2C2:
            return toUniversal(semiDynCast<IGroupModelF2C2<shutter>*>(model)->getCaptureParams(idx));
        case IntriType::kF2:
            return toUniversal(semiDynCast<IGroupModelF2<shutter>*>(model)->getCaptureParams(idx));
        case IntriType::kF1C2D5:
            return toUniversal(semiDynCast<IGroupModelF1C2D5<shutter>*>(model)->getCaptureParams(idx));
        case IntriType::kF1D5:
            return toUniversal(semiDynCast<IGroupModelF1D5<shutter>*>(model)->getCaptureParams(idx));
        case IntriType::kF1D2:
            if (isGrouped()){
                return toUniversal(semiDynCast<IGroupModelF1D2<shutter>*>(model)->getCaptureParams(idx));
            }
            else {
                return toUniversal(semiDynCast<IDiscreteModelF1D2*>(model)->getCaptureParams(idx).pose);
            }
        case IntriType::kF1:
            if (isGrouped()){
                return toUniversal(semiDynCast<IGroupModelF1<shutter>*>(model)->getCaptureParams(idx));
            }
            else {
                return toUniversal(semiDynCast<IDiscreteModelF1*>(model)->getCaptureParams(idx).pose);
            }
    }
    throw std::runtime_error("fatal error");
}

void UniversalModel::setCaptureParams(IdxCap idxCap, const IModel::Pose<true> &params) {
	switch (mModel->getShutterType()) {
#define CASE(x) case ShutterType::x: return setCaptureParamsImpl<ShutterType::x>(idxCap, params)
		CASE(kGlobal);
		CASE(kRollingFixedVelocity);
		CASE(kRolling1D);
		CASE(kRolling1DLoc);
		CASE(kRolling3D);
#undef CASE
	}
	fail();
}

template <ShutterType shutter>
void UniversalModel::setCaptureParamsImpl(IdxCap idxCap, const IModel::Pose<true> &params) {
	constexpr bool rolling = isRolling(shutter);
    IModel* const model = mModel.get();
	require(model->getShutterType() == shutter);
	require(model->hasRollingShutter() == rolling);
    switch (getIntriType()){
        case IntriType::kNull:
            throw std::runtime_error("invalid IntriType");
        case IntriType::kF2C2D5:
            return semiDynCast<IGroupModelF2C2D5<shutter>*>(model)->setCaptureParams(idxCap, fromUniversal<rolling>(params));
        case IntriType::kF2D5:
            return semiDynCast<IGroupModelF2D5<shutter>*>(model)->setCaptureParams(idxCap, fromUniversal<rolling>(params));
        case IntriType::kF2C2:
            return semiDynCast<IGroupModelF2C2<shutter>*>(model)->setCaptureParams(idxCap, fromUniversal<rolling>(params));
        case IntriType::kF2:
            return semiDynCast<IGroupModelF2<shutter>*>(model)->setCaptureParams(idxCap, fromUniversal<rolling>(params));
        case IntriType::kF1C2D5:
            return semiDynCast<IGroupModelF1C2D5<shutter>*>(model)->setCaptureParams(idxCap, fromUniversal<rolling>(params));
        case IntriType::kF1D5:
            return semiDynCast<IGroupModelF1D5<shutter>*>(model)->setCaptureParams(idxCap, fromUniversal<rolling>(params));
        case IntriType::kF1D2:
            if (isGrouped()){
                return semiDynCast<IGroupModelF1D2<shutter>*>(model)->setCaptureParams(idxCap, fromUniversal<rolling>(params));
            }
            else {
				if constexpr (!rolling) {
					const auto m = semiDynCast<IDiscreteModelF1D2*>(model);
					const auto intrinsics = m->getCaptureParams(idxCap).intrinsics;
					assert(std::isnan(m->getCaptureParams(idxCap).pose.q.w));
					m->setCaptureParams(idxCap, {fromUniversal<rolling>(params), intrinsics});
				}
				else {
					fail();
				}
            }
        case IntriType::kF1:
            if (isGrouped()){
                return semiDynCast<IGroupModelF1<shutter>*>(model)->setCaptureParams(idxCap, fromUniversal<rolling>(params));
            }
            else {
				if constexpr (!rolling) {
					const auto m = semiDynCast<IDiscreteModelF1*>(model);
					const auto intrinsics = m->getCaptureParams(idxCap).intrinsics;
					assert(std::isnan(m->getCaptureParams(idxCap).pose.q.w));
					return m->setCaptureParams(idxCap, {fromUniversal<rolling>(params), intrinsics});
				}
				else {
					fail();
				}
            }
    }
    throw std::runtime_error("fatal error");
}

IdxCam UniversalModel::getIdxCamForCapture(IdxCam idxCap) const {
    if (isGrouped()) {
        return semiDynCast<const IGroupModelBase*>(mModel.get())->getIdxCamForCapture(idxCap);
    }
    else {
        return static_cast<IdxCam>(idxCap);
    }
}

bool UniversalModel::isGrouped() const {
    require(mModel->isGrouped() == mIsGrouped);
    return mIsGrouped;
}

} // namespace rba
