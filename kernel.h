//
// Created by yao on 14/09/18.
//

#ifndef RAPIDBA_KERNEL_H
#define RAPIDBA_KERNEL_H
#include <cuda_runtime_api.h>
#include <cstdint>
#include <type_traits>
#include "utils_kernel.h"
#include <utility>
#include "traits/Traits.h"
#include "utils_general.h"
#include "cuda_fp16.h"
#include "csr.h"

namespace rba {

// ftype is the diff type for Coord
template <typename ftype>
struct Point {
    using locf = Coordinate<ftype>;
    locf position[3];
    constexpr static uint32_t DoF = 3;

    template<typename Index>
    __device__ __host__ __forceinline__ locf &operator[](Index i) { return position[i]; }

    template<typename T>
    __device__ __host__ __forceinline__ locf operator[](T i) const { return position[i]; }

    __device__ __host__ __forceinline__
    void update(const kmat<ftype, DoF>& delta) {
        for (int i = 0; i < 3; i++)
            position[i] += delta[i];
    }

    __device__ __host__ __forceinline__
    ftype squaredNorm() const {
        return sqr(position[0].template cast<ftype>()) +
            sqr(position[1].template cast<ftype>()) +
            sqr(position[2].template cast<ftype>());
    }

    bool operator==(const Point& other) const {
        return arrayEqual(position, other.position);
    }
};

// ftype is the diff type for Coord
template <typename ftype>
struct alignas(16) CapOb {
    ftype position[2];
    IdxPt ptIdx;
    half omega;//simplified information matrix
    half huber;
};

// Which points are measured by each capture
template <typename Traits>
using CaptureObservations = rba::VecVec<const CapOb<typename Traits::lpf>, 128, typename Traits::ObIdx>;

template <typename ftype>
struct alignas(8) CtrlLoc
{
    using locf = Coordinate<ftype>;
    locf loc[3];
    ftype huber;
    // 1. For capture, all zero if invalid and all infinity if fixed (hard control point/gnss).
    // 2. Always finite for control points. If you have fixed control point,
    //    use fixed points instead, i.e. remove from model.varPoints.
    symkmat<ftype, 3> omega; // may be quite different from CapOb::omega in orders of magnitude

    __device__ __host__ inline
    bool isHard(int i) const {
        assert(i >= 0 && i < 3);
        return float(omega(i, i)) > std::numeric_limits<float>::max();
    }
    __device__ __host__ inline
    kmat<bool, 3> isHard() const {
        kmat<ftype, 3> diag;
        #pragma once
        for (int i = 0; i < 3; i++) {
            diag[i] = omega(i, i);
        }
        kmat<bool, 3> result;
        #pragma once
        for (int i = 0; i < 3; i++) {
            result[i] = omega(i, i) > std::numeric_limits<float>::max();
        }
        return result;
    }

    static CtrlLoc<ftype> makeDefault() {
        return {{NAN, NAN, NAN}, INFINITY, kmat<ftype, 3, 3>{{0, 0, 0, 0, 0, 0, 0, 0, 0}}};
    }
	__device__ __host__ inline
	bool isInvalid() const {
		return isnan(loc[0].value) || isnan(loc[1].value) || isnan(loc[2].value);
	}
};

template <typename Traits>
struct ModelBase {
    RBA_IMPORT_TRAITS(Traits);
    //varIdx == idxFixed means this point is fixed
    static constexpr uint32_t varIdxFixed = badIdx<uint32_t>();
    struct CamIntrWVarIdx {
        CamIntr intri;
        uint32_t varIdx; //varIdxFixed for fixed intrinsics
    };
    struct alignas(16) PointWVarIdx {
        Point<lpf> pt;
        uint32_t varIdx; //varIdxFixed for fixed point
    };
};
// Which points are viewed from each image
template<typename Traits, bool isConst = false>
struct Model {
    RBA_IMPORT_TRAITS(Traits);
    // not using derivation because want to keep it as POD
    static constexpr uint32_t varIdxFixed = ModelBase<Traits>::varIdxFixed;
    using CamIntrWVarIdx = typename ModelBase<Traits>::CamIntrWVarIdx;
    using PointWVarIdx = typename ModelBase<Traits>::PointWVarIdx;
    using GnssCtrlLoc = CtrlLoc<typename Traits::lpf>;
    using PtCtrlLoc = CtrlLoc<typename Traits::lpf>;

    //@todo: consider adding scale for each variable -> May not be needed as we have pre-conditioner in PCG
    uint32_t nbIntri;
    uint32_t nbCaps;
    uint32_t nbPts;

    typename ConstSelector<CamIntrWVarIdx, isConst>::type *__restrict__ cameras;
    struct {
        const uint32_t* __restrict__ indices;
        const uint32_t nbVar;
    } varCameras;
    typename ConstSelector<Capture, isConst>::type *__restrict__ captures;
    // 1:1 mapped to captures[] if not nullptr. Typically dense or nullptr.
    const GnssCtrlLoc* __restrict__ gnssCtrl;
    struct {
        uint32_t nbVar;
        uint32_t size;
        const uint32_t *__restrict__ indices;//[0, nbVar) are variable and [nbVar, size) are fixed
    } involvedCaps;

    typename ConstSelector<PointWVarIdx, isConst>::type *__restrict__ points;
    struct {
        const uint32_t* __restrict__ indices;
        uint32_t nbVar;
    } varPoints;
    struct {
        // for points[varPoints.indices[idxVarPt[i]]]
        const uint32_t* __restrict__ idxVarPt;
        // Hard control points should use fixed points instead.
        const PtCtrlLoc* __restrict__ data;
        uint32_t size;
    } softCtrlPoints;

    CaptureObservations<Traits> capObs;
};
}
#endif //RAPIDBA_KERNEL_H
