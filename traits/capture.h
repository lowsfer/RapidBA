//
// Created by yao on 9/3/19.
//

#pragma once
#include "../platform.h"
#include <cstdint>
#include "base.h"
#include <cuda_runtime_api.h>
#include "../kmat.h"
#include "../utils_kernel.h"
#include "../utils_general.h"
#include "intrinsic.h"
#include <limits>
#include "../RapidBA.h"

namespace rba{
//using ftype = FPTraitsSSD::lpf;
template <typename ftype_>
struct Pose{
    using ftype = ftype_;
    using locf = Coordinate<ftype>;
    ftype q[4];// w, x, y, z
    locf c[3];// x, y, z
    static constexpr uint32_t DoF = 6;
    static constexpr uint32_t cDofOffset = 3;

    // R = deltaR * R0; C = C0 + deltaC
    __device__ __host__ inline
    void update(const kmat<ftype, DoF>& delta) {
        const ftype deltaGvec[3] = {delta[0], delta[1], delta[2]};
        auto newQ = mat2quat(gvec2mat(kmat<ftype, 3>{deltaGvec}) * quat2mat(kmat<ftype, 4>{q}));
        const auto qScale = fast_rsqrt(newQ.sqrNorm());
        for (int i = 0; i < 4; i++)
            q[i] = newQ[i] * qScale;
        for (int i = 0; i < 3; i++)
            c[i] += delta[3 + i];
        normalize();
    }

    __device__ __host__ inline
    void normalize() {
        const ftype factor = fast_rsqrt(kmat<ftype, 4>{q}.sqrNorm());
        for (auto &e : q)
            e *= factor;
    }

    __device__ __host__ __forceinline__
    ftype squaredNorm() const {
        return 1.f + sqr(c[0].template cast<ftype>()) +
            sqr(c[1].template cast<ftype>()) +
            sqr(c[2].template cast<ftype>());
    }
    bool operator==(const Pose& other) const {
        return arrayEqual(q, other.q) && arrayEqual(c, other.c);
    }

    __device__ __host__ __forceinline__
    const Pose<ftype>& getPose() const {return *this;}
};

template <ShutterType shutter, typename ftype, typename = void>
struct VelocityType;

template <typename ftype>
struct VelocityType<ShutterType::kGlobal, ftype>{
	static constexpr uint32_t DoF = 0;
	static constexpr ShutterType shutter = ShutterType::kGlobal;
	bool operator==(const VelocityType<ShutterType::kGlobal, ftype>& other) const {
        return true;
    }
};
template <typename ftype>
using VelocityNull = VelocityType<ShutterType::kGlobal, ftype>;

// rolling shutter with known flight speed
template <typename ftype_>
struct VelocityType<ShutterType::kRollingFixedVelocity, ftype_> {
	static constexpr ShutterType shutter = ShutterType::kRollingFixedVelocity;
	using ftype = ftype_;
	using locf = Coordinate<ftype>;
	kmat<ftype, 3> velocity; // fixed
	static constexpr uint32_t DoF = 0;
	__device__ __host__ inline
	static VelocityType<shutter, ftype> make(const kmat<ftype, 3>& v) {
		return {v};
	}

	__device__ __host__ inline
    void update(const kmat<ftype, DoF>& delta) {}

    __device__ __host__ __forceinline__
    ftype squaredNorm() const {return 0;}
    bool operator==(const VelocityType<ShutterType::kRollingFixedVelocity, ftype>& other) const {
        return velocity == other.velocity;
    }
	__device__ __host__ __forceinline__
	kmat<ftype, 3> getVelocity() const {
		return velocity;
	}
};
template <typename ftype>
using Velocity0D = VelocityType<ShutterType::kRollingFixedVelocity, ftype>;

// direction of flight speed is known. Only optimize the magnitude
template <ShutterType shutter_, typename ftype_>
struct VelocityType<shutter_, ftype_, std::enable_if_t<shutter_ == ShutterType::kRolling1D || shutter_ == ShutterType::kRolling1DLoc>> {
	using ftype = ftype_;
	using locf = Coordinate<ftype>;
	kmat<ftype, 3> normal; // fixed
	ftype speed;
	static constexpr uint32_t DoF = 1;

	static constexpr ShutterType shutter = shutter_;
	__device__ __host__ inline
	static VelocityType<shutter, ftype> make(const kmat<ftype, 3>& v) {
		const auto mag = std::sqrt(v.sqrNorm());
		const auto scale = 1.f / mag;
		if (std::isfinite(scale)) {
			return {v * scale, mag};
		}
		else {
			return {{0.f, 1.f, 0.f}, 0.f};
		}
	}
	__device__ __host__ inline
    void update(const kmat<ftype, DoF>& delta) {
        speed += delta[0];
    }
    __device__ __host__ __forceinline__
    ftype squaredNorm() const {
        return sqr(speed);
    }
    bool operator==(const VelocityType<shutter, ftype>& other) const {
        return speed == other.speed;
    }
	__device__ __host__ __forceinline__
	kmat<ftype, 3> getVelocity() const {
		return normal * speed;
	}
};
template <typename ftype>
using Velocity1D = VelocityType<ShutterType::kRolling1D, ftype>;
template <typename ftype>
using Velocity1DLoc = VelocityType<ShutterType::kRolling1DLoc, ftype>;

template <typename ftype_>
struct VelocityType<ShutterType::kRolling3D, ftype_>{
	using ftype = float;
	using locf = Coordinate<ftype>;
	static constexpr ShutterType shutter = ShutterType::kRolling3D;
	kmat<ftype, 3> velocity;
	static constexpr uint32_t DoF = 3;
	__device__ __host__ inline
	static VelocityType<shutter, ftype> make(const kmat<ftype, 3>& v) {
		return {v};
	}

	__device__ __host__ inline
    void update(const kmat<ftype, DoF>& delta) {
        velocity = velocity + delta;
    }

    __device__ __host__ __forceinline__
    ftype squaredNorm() const {
        return velocity.sqrNorm();
    }
    bool operator==(const VelocityType<ShutterType::kRolling3D, ftype_>& other) const {
        return velocity == other.velocity;
    }
	__device__ __host__ __forceinline__
	kmat<ftype, 3> getVelocity() const {
		return velocity;
	}
};
template <typename ftype>
using Velocity3D = VelocityType<ShutterType::kRolling3D, ftype>;

template <ShutterType shutterType, typename ftype_>
struct GroupCaptureTemplate {
	static constexpr ShutterType shutter = shutterType;
    using ftype = ftype_;
    using locf = Coordinate<ftype>;
	using Velocity = VelocityType<shutterType, ftype>;

    uint32_t intrIdx;
	Pose<ftype> pose;
	Velocity velocity; // in global coordinate system except ShutterType::kRolling1DLoc

    static constexpr uint32_t DoF = Pose<ftype>::DoF + Velocity::DoF;
    static constexpr uint32_t poseDofOffset = 0;
    static constexpr uint32_t cDofOffset = poseDofOffset + Pose<ftype>::cDofOffset;
	static constexpr uint32_t vDofOffset = poseDofOffset + Pose<ftype>::DoF;

    // R = deltaR * R0; C = C0 + deltaC
    __device__ __host__ inline
    void update(const kmat<ftype, DoF>& delta) {
		pose.update(delta.template block<Pose<ftype>::DoF, 1>(0, 0));
		if constexpr(Velocity::DoF > 0) {
			velocity.update(delta.template block<Velocity::DoF, 1>(vDofOffset, 0u));
		}
    }

    __device__ __host__ __forceinline__
    ftype squaredNorm() const {
		ftype ret = pose.squaredNorm();
		if constexpr(Velocity::DoF > 0) {
			ret += velocity.squaredNorm();
		}
        return ret;
    }
    bool operator==(const GroupCaptureTemplate<shutter, ftype>& other) const {
        return intrIdx == other.intrIdx && pose == other.pose && velocity == other.velocity;
    }
    __device__ __host__ __forceinline__
    const Pose<ftype>& getPose() const {return pose;}
	__device__ __host__ __forceinline__
	kmat<ftype, 3> getVelocity() const {return velocity.getVelocity();}
};

//using ftype = FPTraitsSSD::lpf;
template <typename ftype>
using GroupCaptureGlobal = GroupCaptureTemplate<ShutterType::kGlobal, ftype>;

template <typename ftype>
using GroupCaptureRoll0D = GroupCaptureTemplate<ShutterType::kRollingFixedVelocity, ftype>;

template <typename ftype>
using GroupCaptureRoll1D = GroupCaptureTemplate<ShutterType::kRolling1D, ftype>;

template <typename ftype>
using GroupCaptureRoll1DLoc = GroupCaptureTemplate<ShutterType::kRolling1DLoc, ftype>;

template <typename ftype>
using GroupCaptureRoll3D = GroupCaptureTemplate<ShutterType::kRolling3D, ftype>;

//using ftype = FPTraitsSSD::lpf;
template <typename ftype_>
struct alignas(8) CaptureF1D2 {
    using ftype = ftype_;
    static constexpr uint32_t intrIdx = std::numeric_limits<uint32_t>::max();
    Pose<ftype> pose;// pose.q may be change to rvec to save space and improve alignment
    using Intrinsics = IntrinsicsF1D2<ftype, false>;
    Intrinsics intrinsics;
    static constexpr uint32_t DoF = Pose<ftype>::DoF + Intrinsics::DoF;
    static constexpr uint32_t poseDofOffset = 0;
	using Velocity = VelocityNull<ftype>;

    // R = R0*deltaR; T = T0 + deltaT, not (R0*deltaT + T0)
    __device__ __host__ inline
    void update(const kmat<ftype, DoF>& delta) {
        ftype deltaGvec[Pose<ftype>::DoF];
        for(uint32_t i = 0; i < Pose<ftype>::DoF; i++){
            deltaGvec[i] = delta[i];
        }
        pose.update(deltaGvec);
        kmat<ftype, Intrinsics::DoF> deltaIntri;
        for(uint32_t i = 0; i < Intrinsics::DoF; i++){
            deltaIntri[i] = delta[Pose<ftype>::DoF + i];
        }
        intrinsics.update(deltaIntri);
    }

    __device__ __host__ inline ftype squaredNorm() const {
        return pose.squaredNorm() + intrinsics.squaredNorm();
    }
    bool operator==(const CaptureF1D2& other) const {
        return pose == other.pose && intrinsics == other.intrinsics;
    }
    __device__ __host__ __forceinline__
    const Pose<ftype>& getPose() const {return pose;}
};

template <typename ftype_>
struct alignas(8) CaptureF1 {
    using ftype = ftype_;
    static constexpr uint32_t intrIdx = std::numeric_limits<uint32_t>::max();
    Pose<ftype> pose;// pose.q may be change to rvec to save space and improve alignment
    using Intrinsics = IntrinsicsF1<ftype, false>;
    Intrinsics intrinsics;
    static constexpr uint32_t DoF = Pose<ftype>::DoF + Intrinsics::DoF;
    static constexpr uint32_t poseDofOffset = 0;
	using Velocity = VelocityNull<ftype>;

    // R = R0*deltaR; T = T0 + deltaT, not (R0*deltaT + T0)
    __device__ __host__ inline
    void update(const kmat<ftype, DoF>& delta) {
        ftype deltaGvec[Pose<ftype>::DoF];
        for(uint32_t i = 0; i < Pose<ftype>::DoF; i++){
            deltaGvec[i] = delta[i];
        }
        pose.update(deltaGvec);
        kmat<ftype, Intrinsics::DoF> deltaIntri;
        for(uint32_t i = 0; i < Intrinsics::DoF; i++){
            deltaIntri[i] = delta[Pose<ftype>::DoF + i];
        }
        intrinsics.update(deltaIntri);
    }

    __device__ __host__ inline ftype squaredNorm() const {
        return pose.squaredNorm() + intrinsics.squaredNorm();
    }
    bool operator==(const CaptureF1& other) const {
        return pose == other.pose && intrinsics == other.intrinsics;
    }
    __device__ __host__ __forceinline__
    const Pose<ftype>& getPose() const {return pose;}
};

} // namespace rba
