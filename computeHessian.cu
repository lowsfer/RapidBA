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
// Created by yao on 17/10/18.
//
#include "cuda_hint.cuh"
#include "derivative.h"
#include "kernel.h"
#include "device_launch_parameters.h"
#include <cassert>
#include "csr.h"
#include "utils_general.h"
#include <cooperative_groups.h>
#include "computeHessian.h"
#include "utils_host.h"
#include "utils_kernel.h"
#include <boost/preprocessor/seq/for_each.hpp>
#include <cstdio>
#include "smemAccUtils.h"
namespace cg = cooperative_groups;
namespace rba{

/** structure of Hessian equation
 * | M,  Q,  S |   | deltaC |   | Ec |
 * | Qt, U,  W | * | deltaA | = | Ea |
 * | St, Wt, V |   | deltaB |   | Eb |
 *
 * template <bool isConst = false>
 * struct Hessian;
 *
 * template <bool isConst>
 * struct HessianPreCompData;
 */

struct HessianBlockEnum
{
    static constexpr unsigned M = 1<<0;
    static constexpr unsigned U = 1<<1;
    static constexpr unsigned V = 1<<2;
    static constexpr unsigned Q = 1<<3;
    static constexpr unsigned S = 1<<4;
    static constexpr unsigned W = 1<<5;
    static constexpr unsigned Ec = 1<<6;
    static constexpr unsigned Ea = 1<<7;
    static constexpr unsigned Eb = 1<<8;
    static constexpr unsigned ErrSqrNorm = 1<<9;
    static constexpr unsigned ALL = M|U|V|Q|S|W|Ec|Ea|Eb|ErrSqrNorm;
#define DEFINE_HAS(Name) \
    static constexpr bool has##Name(unsigned tasks) { return (tasks & Name) != 0u; }
    DEFINE_HAS(M)
    DEFINE_HAS(U)
    DEFINE_HAS(V)
    DEFINE_HAS(Q)
    DEFINE_HAS(S)
    DEFINE_HAS(W)
    DEFINE_HAS(Ec)
    DEFINE_HAS(Ea)
    DEFINE_HAS(Eb)
    DEFINE_HAS(ErrSqrNorm)
#undef DEFINE_HAS
};

template <typename Traits, unsigned tasks>
struct HessianSmemAcc
{
    RBA_IMPORT_FPTYPES(Traits);
    using MSymBlock = typename HessianBase<Traits>::MSymBlock;
    using EcBlock = typename HessianBase<Traits>::EcBlock;
    using USymBlock = typename HessianBase<Traits>::USymBlock;
    using EaBlock = typename HessianBase<Traits>::EaBlock;
    using QBlock = typename HessianBase<Traits>::QBlock;

    //@fixme: Maybe it's fine to use float32 for smem accumulators.
#if 0
    symkmat<AccVecType<typename MSymBlock::ValType>, HessianBlockEnum::hasM(tasks) ? MSymBlock::rows() : 0> M;
    kmat<AccVecType<typename EcBlock::ValType>, HessianBlockEnum::hasEc(tasks) ? EcBlock::rows() : 0> Ec;
    symkmat<AccVecType<typename USymBlock::ValType>, HessianBlockEnum::hasU(tasks) ? USymBlock::rows() : 0> U;
    kmat<AccVecType<typename EaBlock::ValType>, HessianBlockEnum::hasEa(tasks) ? EaBlock::rows() : 0> Ea;
    kmat<AccVecType<typename QBlock::ValType>, HessianBlockEnum::hasQ(tasks) ? QBlock::rows() : 0, QBlock::cols()> Q;
    AccVecType<epf> errSqrNorm;
#else
    template <typename T> using AccVecType = smemAccUtils::AccVecType<T>;
    symkmat<AccVecType<float>, HessianBlockEnum::hasM(tasks) ? MSymBlock::rows() : 0> M;
    kmat<AccVecType<float>, HessianBlockEnum::hasEc(tasks) ? EcBlock::rows() : 0> Ec;
    symkmat<AccVecType<typename USymBlock::ValType>, HessianBlockEnum::hasU(tasks) ? USymBlock::rows() : 0> U;
    kmat<AccVecType<typename EaBlock::ValType>, HessianBlockEnum::hasEa(tasks) ? EaBlock::rows() : 0> Ea;
    kmat<AccVecType<typename QBlock::ValType>, HessianBlockEnum::hasQ(tasks) ? QBlock::rows() : 0, QBlock::cols()> Q;
    AccVecType<epf> errSqrNorm;
#endif
};

//For best i-cache usage, split ALL into M, U, Q and V|S|W. @fixme: this seems slower
//constexpr unsigned hessianTask = HessianBlockEnum::ALL;// HessianBlockEnum::V|HessianBlockEnum::W|HessianBlockEnum::S;
//@fixme: shuffle inside warpAtomicAdd seems to be slow. Use 32x (16x for M and Ec) vectorized shared memory accumulators instead.
template <typename Traits, unsigned hessianTask = HessianBlockEnum::ALL>
__global__ void kernel_computeHessian(const Model<Traits, true> model, typename Traits::epf* __restrict__ errSqrNorm,
        const Hessian<Traits, false> hessian, const HessianPreCompData<true> precomp){
    RBA_IMPORT_TRAITS(Traits);
    using GnssCtrlLoc = typename Model<Traits, true>::GnssCtrlLoc;
    using PtCtrlLoc = typename Model<Traits, true>::PtCtrlLoc;
    if (blockIdx.x >= model.involvedCaps.size)
        return;
    assert(blockDim.y == 1 && blockDim.z == 1 && gridDim.y == 1 && gridDim.z == 1);

    const unsigned idxUBlock = blockIdx.x < model.involvedCaps.nbVar ? blockIdx.x : Model<Traits, true>::varIdxFixed;
    const unsigned capIdx = model.involvedCaps.indices[blockIdx.x];

    __shared__ Capture capture;
    __shared__ kmat<bool, 3> hasHardGnss;
    __shared__ typename Model<Traits, true>::CamIntrWVarIdx camera;
    const auto &idxMBlock = camera.varIdx;
    {
        static_assert(sizeof(capture) == sizeof(*model.captures), "fatal error");
        assert(sizeof(Capture) / sizeof(uint32_t) < blockDim.x);
        uint32_t buffer = 0;
        if (threadIdx.x < sizeof(Capture) / sizeof(buffer)) {
            buffer = reinterpret_cast<const uint32_t *>(&model.captures[capIdx])[threadIdx.x];
            reinterpret_cast<uint32_t *>(&capture)[threadIdx.x] = buffer;
        }
        if (threadIdx.x < 3) {
            const auto i = threadIdx.x;
            hasHardGnss[i] = (model.gnssCtrl != nullptr) && model.gnssCtrl[capIdx].isHard(i);
        }

#if 0 // @fixme: change to if constexpr and enable this branch for group model when c++17 is available in device code
        static_assert(sizeof(camera) == sizeof(*model.cameras), "fatal error");
        static_assert(offsetof(Capture, intrIdx) % sizeof(buffer) == 0, "fatal error");
        static_assert(sizeof(camera) < warpSize, "Shuffle-based broadcasting for intrIdx does not work");
        const uint32_t intrIdx = __shfl_sync(0xFFFFFFFFu, buffer, int(offsetof(Capture, intrIdx) / sizeof(buffer)));// @fixme: this actually requires that sizeof(camera)/sizeof(buffer) < warpSize
#else
        if (isGroupModel<Traits>()) {
            __syncthreads();
        }
        const uint32_t intrIdx = capture.intrIdx;
		unused(intrIdx);
#endif
        if (isGroupModel<Traits>()) {
            assert(sizeof(camera) / sizeof(buffer) < blockDim.x);
            if (threadIdx.x < sizeof(camera) / sizeof(buffer)) {
                buffer = reinterpret_cast<const uint32_t *>(&model.cameras[intrIdx])[threadIdx.x];
                reinterpret_cast<uint32_t *>(&camera)[threadIdx.x] = buffer;
            }
        }
		else {
			if (threadIdx.x == 0) {
				camera.varIdx = Model<Traits, true>::varIdxFixed; // otherwise it will try to compute M / Ec blocks
			}
		}
    }
    using SmemAccType = HessianSmemAcc<Traits, hessianTask>;
    __shared__  SmemAccType smemAcc;
    static_assert(sizeof(smemAcc) % sizeof(uint32_t) == 0, "fatal error");
    for (unsigned idx = threadIdx.x; idx < sizeof(smemAcc)/sizeof(uint32_t); idx += blockDim.x)
        reinterpret_cast<uint32_t*>(&smemAcc)[idx] = 0;

    __shared__ const CapOb<lpf> * smemObListCTA;
    __shared__ uint32_t smemNbObsCTA;
    __shared__ const uint32_t* smemPreCompSparseIdxS;
    __shared__ const uint32_t* smemPreCompSparseIdxW;
    //@fixme: replace macro with HessianBlockEnum::hasX()
#define CHECK_HESSIAN_BIT(X) bool(hessianTask & HessianBlockEnum::X)
    if (threadIdx.x == 0){
        smemObListCTA = model.capObs[capIdx];
        smemNbObsCTA = model.capObs.getRowSize(capIdx);
        if (CHECK_HESSIAN_BIT(S)) {
            smemPreCompSparseIdxS = (blockIdx.x < model.involvedCaps.size) ? &precomp.sparseIdxS[precomp.rows[blockIdx.x]] : nullptr;
        }
        if (CHECK_HESSIAN_BIT(W)) {
            smemPreCompSparseIdxW = (blockIdx.x < model.involvedCaps.nbVar) ? &precomp.sparseIdxW[precomp.rows[idxUBlock]] : nullptr;
        }
    }
    __syncthreads();
    if (threadIdx.x < 3 && HessianBlockEnum::hasU(hessianTask)) {
        const auto i = threadIdx.x;
        const bool fixedCapLoc = hasHardGnss[i];
        if (fixedCapLoc) {
            assert(model.gnssCtrl[capIdx].loc[i] == capture.pose.c[i]);
            // use atomic so we don't need another sync.
            const auto idxDof = Capture::poseDofOffset + decltype(capture.pose)::cDofOffset + i;
            assert(model.gnssCtrl[capIdx].loc[i] == capture.pose.c[i]);
            smemAcc.U(idxDof, idxDof).atomicRed(1.f);
        }
        // soft GNSS is handled by a dedicated kernel.
    }

    const CapOb<lpf>* const& obListCTA = smemObListCTA;
    const uint32_t& nbObsCTA = smemNbObsCTA;
    const uint32_t*& sparseIdxSTable = smemPreCompSparseIdxS;
    const uint32_t*& sparseIdxWTable = smemPreCompSparseIdxW;

    CapOb<lpf> obPrePre;//two levels of preload
    {
        const unsigned idxObLocal = threadIdx.x;
        if (idxObLocal < nbObsCTA)
            obPrePre = ldg(&obListCTA[idxObLocal]);
    }
    CapOb<lpf> obPre;
#ifndef NDEBUG
    obPre.ptIdx = 0xFFFFFFFF;
#endif
    typename Model<Traits, true>::PointWVarIdx ptPre;
    auto preload = [&](unsigned idxObLocal) {
        if (idxObLocal < nbObsCTA) {
            obPre = obPrePre;
            ptPre = model.points[obPre.ptIdx];
            if (idxObLocal + blockDim.x < nbObsCTA)
                obPrePre = ldg(&obListCTA[idxObLocal + blockDim.x]);
        }
    };
    preload(threadIdx.x);
    for (unsigned idxObLocal = threadIdx.x; idxObLocal < nbObsCTA; idxObLocal += blockDim.x) {
        const CapOb<lpf> observation = obPre;
        const Point<lpf> pt = ptPre.pt;
        const uint32_t idxVBlock = ptPre.varIdx;
        preload(idxObLocal + blockDim.x);
        const auto errDer = computeErrorDerivative<Traits>(camera.intri, capture,
            kmat<locf, Point<lpf>::DoF>{pt.position}, observation.position,
            hasHardGnss);
        lpf robustChi2 = NAN;
        const lpf weightedOmega = robustify<lpf>(errDer.error, observation.omega, observation.huber, &robustChi2);
#if 0
        if (!std::isfinite(robustChi2)) {
            if constexpr(std::is_same_v<Traits, TraitsGrpF1D5Global>) {
                printf("robustChi2 = %f, error = (%f, %f), f=%f, d=[%f,%f,%f,%f,%f], pose=(%f,%f,%f,%f, %f,%f,%f), pt=(%f,%f,%f), idxMBlock=%u\n", robustChi2, errDer.error[0], errDer.error[1],
                    camera.intri.f,
                    camera.intri.d.params[0], camera.intri.d.params[1], camera.intri.d.params[2], camera.intri.d.params[3], camera.intri.d.params[4],
                    capture.pose.q[0], capture.pose.q[1], capture.pose.q[2], capture.pose.q[3],
                    capture.pose.c[0].value, capture.pose.c[1].value, capture.pose.c[2].value,
                    pt.position[0].value, pt.position[1].value, pt.position[2].value,
                    idxMBlock
                );
            }
        }
#endif
        // ErrSqrNorm
        if (CHECK_HESSIAN_BIT(ErrSqrNorm)){
            smemAcc.errSqrNorm.atomicRed(robustChi2);
        }
        // M and Ec
        if ((CHECK_HESSIAN_BIT(M) || CHECK_HESSIAN_BIT(Ec)) && idxMBlock != Model<Traits, true>::varIdxFixed)
        {
            assert(model.cameras[model.varCameras.indices[idxMBlock]].varIdx == idxMBlock);
            const symkmat<lpf, CamIntr::DoF> M =
                    symkmat<lpf, CamIntr::DoF>(errDer.jacobian.camera.transpose() * weightedOmega * errDer.jacobian.camera);
            const kmat<lpf, CamIntr::DoF> Ec =
                    errDer.jacobian.camera.transpose() * (weightedOmega * errDer.error);
            if (CHECK_HESSIAN_BIT(M)) {
                for (unsigned idx = 0; idx < M.size(); idx++) {
                    smemAcc.M.data()[idx].atomicRed(M.data()[idx]);
                }
            }
            if (CHECK_HESSIAN_BIT(Ec)) {
                for (unsigned idx = 0; idx < Ec.size(); idx++) {
                    smemAcc.Ec[idx].atomicRed(Ec[idx]);
                }
            }
        }
        // U and Ea
        if ((CHECK_HESSIAN_BIT(U) || CHECK_HESSIAN_BIT(Ea)) && idxUBlock != Model<Traits, true>::varIdxFixed)
        {
            const symkmat<lpf, Capture::DoF> U =
                    symkmat<lpf, Capture::DoF>(errDer.jacobian.capture.transpose() * weightedOmega * errDer.jacobian.capture);
#if RBA_DBG_PRINT_OBSERVATION
            if (idxUBlock == 72) {
                if (idxObLocal == 0){
                    const auto intr = getIntrinsics(camera.intri, capture);
                    const auto pose = capture.getPose();
                    printf("q = [%f, %f, %f,%f], t = [%f, %f, %f], f = %f, k1 = %f, k2 = %f, pt = [%f, %f, %f], proj = [%f, %f]\n",
                           pose.q[0], pose.q[1], pose.q[2], pose.q[3], pose.t[0], pose.t[1], pose.t[2], intr.f, intr.d.params[0], intr.d.params[1], pt.position[0], pt.position[1], pt.position[2], observation.position[0], observation.position[1]);
                }
            }
#endif
            const kmat<lpf, Capture::DoF> Ea =
                    errDer.jacobian.capture.transpose() * (weightedOmega * errDer.error);
            if (CHECK_HESSIAN_BIT(U)) {
                for (unsigned idx = 0; idx < U.size(); idx++) {
                    smemAcc.U.data()[idx].atomicRed(U.data()[idx]);
                }
            }
            if (CHECK_HESSIAN_BIT(Ea)) {
                for (unsigned idx = 0; idx < Ea.size(); idx++) {
                    smemAcc.Ea[idx].atomicRed(Ea[idx]);
                }
            }
        }
        // V and Eb
        if((CHECK_HESSIAN_BIT(V) || CHECK_HESSIAN_BIT(Eb)) && idxVBlock != Model<Traits, true>::varIdxFixed)
        {
            assert(model.points[model.varPoints.indices[idxVBlock]].varIdx == idxVBlock);
            const symkmat<lpf, Point<lpf>::DoF> V =
                    symkmat<lpf, Point<lpf>::DoF>(errDer.jacobian.pt.transpose() * weightedOmega * errDer.jacobian.pt);
            const kmat<lpf, Point<lpf>::DoF> Eb = errDer.jacobian.pt.transpose() * (weightedOmega * errDer.error);
            // @fixme: if atomicAdd is bottleneck, transpose first in shared memory
            if (CHECK_HESSIAN_BIT(V)) {
                for (unsigned idx = 0; idx < V.size(); idx++) {
                    atomicAdd(&hessian.V[idxVBlock].data()[idx], V.data()[idx]);
                }
            }
            if (CHECK_HESSIAN_BIT(Eb)) {
                for (unsigned idx = 0; idx < Eb.size(); idx++) {
                    atomicAdd(&hessian.Eb[idxVBlock][idx], Eb[idx]);
                }
            }
        }
        // Q
        assert((idxUBlock == Model<Traits, true>::varIdxFixed) || hessian.Q.row == nullptr || hessian.Q.row[idxUBlock] == idxMBlock);
        if (CHECK_HESSIAN_BIT(Q) && idxMBlock != Model<Traits, true>::varIdxFixed && idxUBlock != Model<Traits, true>::varIdxFixed)
        {
            assert(model.cameras[model.varCameras.indices[idxMBlock]].varIdx == idxMBlock);
            assert(hessian.Q.row[idxUBlock] == idxMBlock);
            const kmat<lpf, CamIntr::DoF, Capture::DoF> Q = errDer.jacobian.camera.transpose() * (weightedOmega * errDer.jacobian.capture);
            for (unsigned idx = 0; idx < Q.size(); idx++) {
                smemAcc.Q.data()[idx].atomicRed(Q.data()[idx]);
            }
        }
        // S
        if (CHECK_HESSIAN_BIT(S) && idxMBlock != Model<Traits, true>::varIdxFixed && idxVBlock != Model<Traits, true>::varIdxFixed)
        {
            assert(model.cameras[model.varCameras.indices[idxMBlock]].varIdx == idxMBlock);
            assert(model.points[model.varPoints.indices[idxVBlock]].varIdx == idxVBlock);
            const kmat<lpf, CamIntr::DoF, Point<lpf>::DoF> S = errDer.jacobian.camera.transpose() * (weightedOmega * errDer.jacobian.pt);
//            const uint32_t idxS = hessian.S.rows[idxMBlock] + bisect(&hessian.S.idxCol[hessian.S.rows[idxMBlock]], 0, hessian.S.rows[idxMBlock+1] - hessian.S.rows[idxMBlock], idxVBlock);
            // precompute and put the bisect results in a LUT
            assert(sparseIdxSTable[idxObLocal] != ModelBase<Traits>::varIdxFixed);
            const uint32_t idxS = ldg(&hessian.S.rows[idxMBlock]) + ldg(&sparseIdxSTable[idxObLocal]);
            assert(hessian.S.idxCol[idxS] == idxVBlock);
            // @fixme: if atomicAdd is bottleneck, transpose first in shared memory
            for (unsigned idx = 0; idx < S.size(); idx++) {
                atomicAdd(&hessian.S.data[idxS].data()[idx], S.data()[idx]);
            }
        }
        // W
        if (CHECK_HESSIAN_BIT(W) && idxUBlock != Model<Traits, true>::varIdxFixed && idxVBlock != Model<Traits, true>::varIdxFixed)
        {
            assert(model.points[model.varPoints.indices[idxVBlock]].varIdx == idxVBlock);
            const kmat<lpf, Capture::DoF, Point<lpf>::DoF> W = errDer.jacobian.capture.transpose() * (weightedOmega * errDer.jacobian.pt);
            assert(sparseIdxWTable[idxObLocal] != ModelBase<Traits>::varIdxFixed);
            // sparseIdxWTable is used because some observations are fixed and does not map to W 1:1.
            const uint32_t idxW = ldg(&hessian.W.rows[idxUBlock]) + ldg(&sparseIdxWTable[idxObLocal]);
            assert(hessian.W.idxCol[idxW] == idxVBlock);
            // @fixme: if STG is bottleneck, transpose first in shared memory
            hessian.W.data[idxW] = W;
        }
    }

    __syncthreads();
    {
        if ((CHECK_HESSIAN_BIT(M) || CHECK_HESSIAN_BIT(Ec)) && idxMBlock != Model<Traits, true>::varIdxFixed) {
            if (CHECK_HESSIAN_BIT(M)) {
                using MType = std::decay_t<decltype(hessian.M[0])>;
                smemAccUtils::ctaRed<MType::size(), typename MType::ValType, typename decltype(smemAcc.M)::ValType::ValType, true>(
                        hessian.M[idxMBlock].data(), smemAcc.M.data());
            }
            if (CHECK_HESSIAN_BIT(Ec)) {
                using EcType = std::decay_t<decltype(hessian.Ec[0])>;
                smemAccUtils::ctaRed<EcType::size(), typename EcType::ValType, typename decltype(smemAcc.Ec)::ValType::ValType, true>(
                        hessian.Ec[idxMBlock].data(), smemAcc.Ec.data());
            }
        }
        if ((CHECK_HESSIAN_BIT(U) || CHECK_HESSIAN_BIT(Ea)) && idxUBlock != Model<Traits, true>::varIdxFixed) {
            if (CHECK_HESSIAN_BIT(U)) {
                using UType = std::decay_t<decltype(hessian.U[0])>;
                smemAccUtils::ctaRed<UType::size(), typename UType::ValType, typename decltype(smemAcc.U)::ValType::ValType, false>(
                        hessian.U[idxUBlock].data(), smemAcc.U.data());
            }
            if (CHECK_HESSIAN_BIT(Ea)) {
                using EaType = std::decay_t<decltype(hessian.Ea[0])>;
                smemAccUtils::ctaRed<EaType::size(), typename EaType::ValType, typename decltype(smemAcc.Ea)::ValType::ValType, false>(
                        hessian.Ea[idxUBlock].data(), smemAcc.Ea.data());
            }
        }
        if (CHECK_HESSIAN_BIT(Q) && idxMBlock != Model<Traits, true>::varIdxFixed && idxUBlock != Model<Traits, true>::varIdxFixed)
        {
            assert(hessian.Q.row[idxUBlock] == idxMBlock);
            using QType = std::decay_t<decltype(hessian.Q.blocks[0])>;
            smemAccUtils::ctaRed<QType::size(), typename QType::ValType, typename decltype(smemAcc.Q)::ValType::ValType, false>(
                    hessian.Q.blocks[idxUBlock].data(), smemAcc.Q.data());
        }
        //@fixme: There is no test for this added errSeqNorm computation
        if (CHECK_HESSIAN_BIT(ErrSqrNorm)){
            using ErrSqrNormType = std::decay_t<decltype(errSqrNorm[0])>;
            smemAccUtils::ctaRed<1u, ErrSqrNormType, typename decltype(smemAcc.errSqrNorm)::ValType, true>(errSqrNorm, &smemAcc.errSqrNorm);
        }
    }
#undef CHECK_HESSIAN_BIT
}

template <typename Traits>
__global__ void kernel_updateHessianWithSoftCtrlPts(const Model<Traits, true> model, typename Traits::epf* __restrict__ errSqrNorm,
    const Hessian<Traits, false> hessian)
{
    using lpf = typename Traits::lpf;
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= model.softCtrlPoints.size) {
        return;
    }
    const auto idxVarPt = model.softCtrlPoints.idxVarPt[tid];
    assert(idxVarPt < model.varPoints.nbVar);
    const auto idxPt = model.varPoints.indices[idxVarPt];
    const CtrlLoc<lpf> ctrlPt = model.softCtrlPoints.data[tid];

    assert(model.points[idxPt].varIdx == idxVarPt);
    const Point<lpf> pt = model.points[idxPt].pt;
    assert(!ctrlPt.isHard(0) && !ctrlPt.isHard(1) && !ctrlPt.isHard(2));
    const bool updateErrSqrNorm = (errSqrNorm != nullptr);
    const kmat<lpf, Point<lpf>::DoF> error = computeCtrlError<lpf>(pt.position, ctrlPt.loc);
    lpf robustChi2 = NAN;
    const auto weightedOmega = robustify<lpf>(error, ctrlPt.omega, ctrlPt.huber, &robustChi2);
    #pragma unroll
    for (int i = 0; i < weightedOmega.size(); i++) {
        atomicAdd(&hessian.V[idxVarPt].data()[i], weightedOmega.data()[i]); // jacobian is eye(3)
    }
    const kmat<lpf, Point<lpf>::DoF> Eb = weightedOmega.toKMat() * error;
    #pragma unroll
    for (int i = 0; i < Point<lpf>::DoF; i++) {
        atomicAdd(&hessian.Eb[idxVarPt][i], Eb[i]);
    }
    
    if (updateErrSqrNorm) {
        atomicAdd(errSqrNorm, robustChi2);
    }
}

template <typename Traits>
__global__ void kernel_updateHessianWithSoftGNSS(const Model<Traits, true> model, typename Traits::epf* __restrict__ errSqrNorm,
    const Hessian<Traits, false> hessian)
{
    using lpf = typename Traits::lpf;
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t idxVarCap = tid;
    if (idxVarCap >= model.involvedCaps.nbVar) {
        return;
    }
    const uint32_t idxCap = model.involvedCaps.indices[idxVarCap];
    const auto gnssCtrl = model.gnssCtrl[idxCap];
	if (gnssCtrl.isInvalid()){
		return;
	}
    const kmat<bool, 3> isHard = gnssCtrl.isHard();
    if (isHard[0] && isHard[1] && isHard[2]) {
        return;
    }
    const kmat<Coordinate<lpf>, 3> capLoc {model.captures[idxCap].pose.c};
    const kmat<lpf, 3> error = computeCtrlError<lpf>(capLoc, gnssCtrl.loc);
    lpf robustChi2 = NAN;
    const symkmat<lpf, 3> weightedOmega = robustify<lpf>(error, gnssCtrl.omega, gnssCtrl.huber, &robustChi2);
    const kmat<lpf, 3> Ea = weightedOmega.toKMat() * error;
    const symkmat<lpf, 3> Uloc = weightedOmega.toKMat();

    using Capture = typename Traits::Capture;
    constexpr auto offset = Capture::poseDofOffset + decltype(std::declval<Capture>().pose)::cDofOffset;
    #pragma unroll
    for (int i = 0; i < Point<lpf>::DoF; i++) {
        if (!isHard[i]) {
            for (int j = i; j < Point<lpf>::DoF; j++) {
                if (!isHard[j]) {
                    atomicAdd(&hessian.U[idxVarCap](offset + i, offset + j), Uloc(i, j));
                }
            }
            atomicAdd(&hessian.Ea[idxVarCap][offset + i], Ea[i]);
        }
    }
    if (errSqrNorm != nullptr) {
        atomicAdd(errSqrNorm, robustChi2);
    }
}

#if 0 && __CUDACC_VER_MAJOR__ >= 10
template <typename Traits>
UniqueCudaGraph cudaComputeHessian(Model<Traits, true> model, HessianPreCompData<true> precomp, Hessian<Traits, false> hessian){
    auto graph = makeCudaGraph();
    auto makeParams = [&](void* func){
        dim3 dimBlock(128);
        dim3 dimGrid(model.involvedCaps.size);
        void* kernelArgs[] = {&model, &precomp, &hessian};
        return cudaKernelNodeParams {
            func,
            dimGrid,
            dimBlock,
            0,
            kernelArgs,
            nullptr
        };
    };

    auto params = makeParams((void*)&kernel_computeHessian<Traits, HessianBlockEnum::M>);
    checkCudaErrors(cudaGraphAddKernelNode(nullptr, graph.get(), nullptr, 0, &params));

    params = makeParams((void*)&kernel_computeHessian<Traits, HessianBlockEnum::U>);
    checkCudaErrors(cudaGraphAddKernelNode(nullptr, graph.get(), nullptr, 0, &params));

    params = makeParams((void*)&kernel_computeHessian<Traits, HessianBlockEnum::Q>);
    checkCudaErrors(cudaGraphAddKernelNode(nullptr, graph.get(), nullptr, 0, &params));

    params = makeParams((void*)&kernel_computeHessian<Traits, HessianBlockEnum::V | HessianBlockEnum::S | HessianBlockEnum::W>);
    checkCudaErrors(cudaGraphAddKernelNode(nullptr, graph.get(), nullptr, 0, &params));

    return graph;
}
#endif

//@fixme: there is no test for this
template <typename Traits>
cudaError_t launchCudaComputeErrSqrNorm(const Model<Traits, true>& model, double* errSqrNorm, cudaStream_t stream) {
    if (model.involvedCaps.size == 0)
        return cudaSuccess;

    checkEarlyReturn(cudaMemsetAsync(errSqrNorm, 0, sizeof(*errSqrNorm), stream));

    dim3 dimBlock(128);
    dim3 dimGrid(model.involvedCaps.size);

    return launchKernel(kernel_computeHessian<Traits, HessianBlockEnum::ErrSqrNorm>,
            dimGrid, dimBlock, 0, stream,
            model, errSqrNorm, Hessian<Traits, false>{}, HessianPreCompData<true>{});
}

#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchCudaComputeErrSqrNorm(const Model<TRAITS, true>& model, double* errSqrNorm, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES
#if 0
//@fixme: this (use separate kernels for differnet blocks) is slower than the other approach using one single kernel. Fix possible corner cases (nbVar* == 0) for that code and use that instead
template <typename Traits>
cudaError_t launchCudaComputeHessian(const Model<Traits, true>& model, double* errSqrNorm, const Hessian<Traits, false>& hessian, const uint32_t nbSBlocks, const HessianPreCompData<true>& precomp, cudaStream_t stream){
    if (model.involvedCaps.size == 0)
        return cudaSuccess;

    if (errSqrNorm!= nullptr) {
        checkEarlyReturn(cudaMemsetAsync(errSqrNorm, 0, sizeof(*errSqrNorm), stream));
    }

    dim3 dimBlock(128);
    dim3 dimGrid(model.involvedCaps.size);

#ifndef NDEBUG
    checkCudaErrors(cudaMemsetAsync(hessian.U, 0, sizeof(hessian.U[0]) * hessian.nbVarCaps, stream));
    checkCudaErrors(cudaMemsetAsync(hessian.Ea, 0, sizeof(hessian.Ea[0]) * hessian.nbVarCaps, stream));
    checkCudaErrors(cudaMemsetAsync(hessian.Q.blocks, 0, sizeof(hessian.Q.blocks[0]) * hessian.nbVarCaps, stream));
#endif

    if (hessian.nbVarIntri != 0) {
        checkEarlyReturn(cudaMemsetAsync(hessian.M, 0, sizeof(hessian.M[0]) * hessian.nbVarIntri, stream));
        checkEarlyReturn(cudaMemsetAsync(hessian.Ec, 0, sizeof(hessian.Ec[0]) * hessian.nbVarIntri, stream));
        checkEarlyReturn(launchKernel(kernel_computeHessian<Traits, HessianBlockEnum::M | HessianBlockEnum::Ec>,
                dimGrid, dimBlock, 0, stream,
                model, errSqrNorm, hessian, precomp));
    }
    {
        checkEarlyReturn(launchKernel(kernel_computeHessian<Traits, HessianBlockEnum::U | HessianBlockEnum::Ea>,
                dimGrid, dimBlock, 0, stream,
                model, errSqrNorm, hessian, precomp));
    }
    if (hessian.nbVarIntri != 0 && hessian.nbVarCaps != 0) {
        checkEarlyReturn(launchKernel(kernel_computeHessian<Traits, HessianBlockEnum::Q>, dimGrid, dimBlock, 0, stream, model, errSqrNorm, hessian, precomp));
    }
    if (hessian.nbVarPts != 0) {
        checkEarlyReturn(cudaMemsetAsync(hessian.V, 0, sizeof(hessian.V[0]) * hessian.nbVarPts, stream));
        checkEarlyReturn(cudaMemsetAsync(hessian.Eb, 0, sizeof(hessian.Eb[0]) * hessian.nbVarPts, stream));
        assert(nbSBlocks == hessian.S.nbElems());
        checkEarlyReturn(cudaMemsetAsync(hessian.S.data, 0, sizeof(hessian.S.data[0]) * nbSBlocks, stream));
#ifndef NDEBUG
        checkCudaErrors(cudaMemsetAsync(hessian.W.data, 0, sizeof(hessian.W.data[0]) * hessian.W.nbElems(), stream));
#endif
        if (errSqrNorm != nullptr) {
            checkEarlyReturn(launchKernel(
                    kernel_computeHessian<Traits, HessianBlockEnum::V | HessianBlockEnum::Eb | HessianBlockEnum::S | HessianBlockEnum::W | HessianBlockEnum::ErrSqrNorm>,
                    dimGrid, dimBlock, 0, stream, model, errSqrNorm, hessian, precomp));
        }
        else{
            checkEarlyReturn(launchKernel(
                    kernel_computeHessian<Traits, HessianBlockEnum::V | HessianBlockEnum::Eb | HessianBlockEnum::S | HessianBlockEnum::W>,
                    dimGrid, dimBlock, 0, stream, model, errSqrNorm, hessian, precomp));
        }
    }
    else if(errSqrNorm != nullptr) {
        checkEarlyReturn(launchCudaComputeErrSqrNorm(model, errSqrNorm, stream));
    }
    return cudaSuccess;
}
#else //@fixme: looks like this is faster. May need more tests for corner cases with nbVar* == 0
template <typename Traits>
cudaError_t launchCudaComputeHessian(const Model<Traits, true>& model, double* errSqrNorm, const Hessian<Traits, false>& hessian, const uint32_t nbSBlocks, const HessianPreCompData<true>& precomp, cudaStream_t stream){
    if (model.involvedCaps.size == 0)
        return cudaSuccess;

    if (errSqrNorm!= nullptr) {
        checkEarlyReturn(cudaMemsetAsync(errSqrNorm, 0, sizeof(*errSqrNorm), stream));
    }

#ifndef NDEBUG
    checkCudaErrors(cudaMemsetAsync(hessian.U, 0, sizeof(hessian.U[0]) * hessian.nbVarCaps, stream));
    checkCudaErrors(cudaMemsetAsync(hessian.Ea, 0, sizeof(hessian.Ea[0]) * hessian.nbVarCaps, stream));
    checkCudaErrors(cudaMemsetAsync(hessian.Q.blocks, 0, sizeof(hessian.Q.blocks[0]) * hessian.nbVarCaps, stream));
#endif

    if (hessian.nbVarIntri != 0) {
        checkEarlyReturn(cudaMemsetAsync(hessian.M, 0, sizeof(hessian.M[0]) * hessian.nbVarIntri, stream));
        checkEarlyReturn(cudaMemsetAsync(hessian.Ec, 0, sizeof(hessian.Ec[0]) * hessian.nbVarIntri, stream));
    }
    if (hessian.nbVarPts != 0) {
        checkEarlyReturn(cudaMemsetAsync(hessian.V, 0, sizeof(hessian.V[0]) * hessian.nbVarPts, stream));
        if (model.softCtrlPoints.size != 0) {
            checkEarlyReturn(launchKernel(kernel_updateHessianWithSoftCtrlPts<Traits>,
                dim3(divUp(model.softCtrlPoints.size, 128u)), dim3(128), 0, stream,
                model, errSqrNorm, hessian));
        }
        checkEarlyReturn(cudaMemsetAsync(hessian.Eb, 0, sizeof(hessian.Eb[0]) * hessian.nbVarPts, stream));
        assert(nbSBlocks == hessian.S.nbElems());
        checkEarlyReturn(cudaMemsetAsync(hessian.S.data, 0, sizeof(hessian.S.data[0]) * nbSBlocks, stream));
#ifndef NDEBUG
        checkCudaErrors(cudaMemsetAsync(hessian.W.data, 0, sizeof(hessian.W.data[0]) * hessian.W.nbElems(), stream));
#endif
    }

    dim3 dimBlock(128);
    dim3 dimGrid(model.involvedCaps.size);
    if(errSqrNorm == nullptr) {
        if (hessian.W.data != nullptr) {
            checkEarlyReturn(launchKernel(
                    kernel_computeHessian < Traits, HessianBlockEnum::ALL & ~HessianBlockEnum::ErrSqrNorm > ,
                    dimGrid, dimBlock, 0, stream, model, errSqrNorm, hessian, precomp));
        }
        else{
            checkEarlyReturn(launchKernel(
                    kernel_computeHessian < Traits, HessianBlockEnum::ALL & ~HessianBlockEnum::W & ~HessianBlockEnum::ErrSqrNorm > ,
                    dimGrid, dimBlock, 0, stream, model, errSqrNorm, hessian, precomp));
        }
    }else{
        if (hessian.W.data != nullptr) {
            checkEarlyReturn(launchKernel(
                    kernel_computeHessian < Traits, HessianBlockEnum::ALL > ,
                    dimGrid, dimBlock, 0, stream, model, errSqrNorm, hessian, precomp));
        }
        else {
            checkEarlyReturn(launchKernel(
                    kernel_computeHessian<Traits, HessianBlockEnum::ALL  & ~HessianBlockEnum::W>,
                    dimGrid, dimBlock, 0, stream, model, errSqrNorm, hessian, precomp));
        }
    }
    if (model.gnssCtrl != nullptr) {
        checkEarlyReturn(launchKernel(&kernel_updateHessianWithSoftGNSS<Traits>,
            dim3(divUp(int(model.involvedCaps.nbVar), 128)), dim3(128), 0, stream,
            model, errSqrNorm, hessian));
    }
    return cudaSuccess;
}
#endif
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchCudaComputeHessian(const Model<TRAITS, true>& model, double* errSqrNorm, const Hessian<TRAITS, false>& hessian, const uint32_t nbSBlocks, const HessianPreCompData<true>& precomp, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

namespace dampLM {
constexpr uint32_t ctaSize = 128;

__device__ __host__ inline
void makeCtaIdxEnd(uint32_t (&idxCtaEnd)[3], uint32_t nbVarIntri, uint32_t nbVarCaps, uint32_t nbVarPts){
    const uint32_t nbCtas[3] = {
            divUp(nbVarIntri, ctaSize),
            divUp(nbVarCaps, ctaSize),
            divUp(nbVarPts, ctaSize)//unused
    };
    idxCtaEnd[0] = nbCtas[0];
    for (int i = 1; i < 3; i++)
        idxCtaEnd[i] = idxCtaEnd[i-1] + nbCtas[i];
}

template <typename Traits, bool isConst>
__device__ __host__ inline
void makeCtaIdxEnd(uint32_t (&idxCtaEnd)[3], const Hessian<Traits, isConst>& hessian){
    makeCtaIdxEnd(idxCtaEnd, hessian.nbVarIntri, hessian.nbVarCaps, hessian.nbVarPts);
}

template <typename T>
__device__ __host__ T getDampMu(T val, T lambda){
//    return lambda; // SBA approach
//    return lambda * std::sqrt(val); // PBA approach
    return lambda * std::max(T(1E-6f), std::sqrt(val));
}

template <typename T>
__device__ __host__ T damp(T val, T lambda)
{
    return val + getDampMu(val, lambda);
}

template <typename Traits>
__global__ void kernel_backupAndDamp(HessianVec<Traits, false> backup, Hessian<Traits, false> hessian, float lambda) {
    uint32_t idxCtaEnd[3];
    makeCtaIdxEnd(idxCtaEnd, hessian);
    assert(gridDim.x == idxCtaEnd[2]);
    const uint32_t idxCta = blockIdx.x;
    auto exec = [lambda](auto* vectors, auto* blocks, uint32_t idx, uint32_t idxMax)->void{
        assert(vectors[0].rows() == blocks[0].rows());
        if (idx >= idxMax)
            return;
        using VecType = typename std::remove_pointer<decltype(vectors)>::type;
        VecType vec;
        auto& block = blocks[idx];
        for (uint32_t i = 0; i < vec.size(); i++)
            vec[i] = block(i, i);
        for (uint32_t i = 0; i < vec.size(); i++) {
            assert(vec[i] > 0);
            block(i, i) = damp<std::decay_t<decltype(vec[i])>>(vec[i], lambda);
        }
        vectors[idx] = vec;
    };
    if (idxCta < idxCtaEnd[0]) {
        const uint32_t idx = ctaSize * idxCta + threadIdx.x;
        exec(backup.c, hessian.M, idx, backup.nbCBlocks);
    }
    else if (idxCta < idxCtaEnd[1]) {
        const uint32_t idx = ctaSize * (idxCta - idxCtaEnd[0]) + threadIdx.x;
        exec(backup.a, hessian.U, idx, backup.nbABlocks);
    }
    else {
        assert(idxCta < idxCtaEnd[2]);
        const uint32_t idx = ctaSize * (idxCta - idxCtaEnd[1]) + threadIdx.x;
        exec(backup.b, hessian.V, idx, backup.nbBBlocks);
    }
}

template <typename Traits>
cudaError_t launchCudaBackupAndDamp(HessianVec<Traits, false> backup, Hessian<Traits, false> hessian, float damp, cudaStream_t stream){
    uint32_t idxCtaEnd[3];
    makeCtaIdxEnd(idxCtaEnd, hessian);
    return launchKernel(kernel_backupAndDamp<Traits>, idxCtaEnd[2], ctaSize, 0, stream, backup, hessian, damp);
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchCudaBackupAndDamp(HessianVec<TRAITS, false> backup, Hessian<TRAITS, false> hessian, float damp, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES


template <typename Traits>
__global__ void kernel_dampFromBackup(HessianVec<Traits, true> backup, Hessian<Traits, false> hessian, float lambda) {
    uint32_t idxCtaEnd[3];
    makeCtaIdxEnd(idxCtaEnd, hessian);
    assert(gridDim.x == idxCtaEnd[2]);
    const uint32_t idxCta = blockIdx.x;
    auto exec = [lambda](const auto* vectors, auto* blocks, uint32_t idx, uint32_t idxMax)->void{
        if (idx >= idxMax)
            return;
        const auto vec = vectors[idx];
        auto& block = blocks[idx];
        for (uint32_t i = 0; i < vec.size(); i++) {
            assert(vec[i] >= 0);
            block(i, i) = damp<std::decay_t<decltype(vec[i])>>(vec[i], lambda);
        }
    };
    if (idxCta < idxCtaEnd[0]) {
        const uint32_t idx = ctaSize * idxCta + threadIdx.x;
        exec(backup.c, hessian.M, idx, backup.nbCBlocks);
    }
    else if (idxCta < idxCtaEnd[1]) {
        const uint32_t idx = ctaSize * (idxCta - idxCtaEnd[0]) + threadIdx.x;
        exec(backup.a, hessian.U, idx, backup.nbABlocks);
    }
    else {
        assert(idxCta < idxCtaEnd[2]);
        const uint32_t idx = ctaSize * (idxCta - idxCtaEnd[1]) + threadIdx.x;
        exec(backup.b, hessian.V, idx, backup.nbBBlocks);
    }
}

template <typename Traits>
cudaError_t launchCudaDampFromBackup(HessianVec<Traits, true> backup, Hessian<Traits, false> hessian, float damp, cudaStream_t stream){
    uint32_t idxCtaEnd[3];
    makeCtaIdxEnd(idxCtaEnd, hessian);
    return launchKernel(kernel_dampFromBackup<Traits>, idxCtaEnd[2], ctaSize, 0, stream, backup, hessian, damp);
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchCudaDampFromBackup(HessianVec<TRAITS, true> backup, Hessian<TRAITS, false> hessian, float damp, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template <typename Traits, bool isConst>
__device__ __host__ inline
void makeCtaIdxEnd(uint32_t (&idxCtaEnd)[3], const HessianVec<Traits, isConst>& vec){
    makeCtaIdxEnd(idxCtaEnd, vec.nbCBlocks, vec.nbABlocks, vec.nbBBlocks);
}

template <typename Traits>
__global__ void kernel_computeExpectedErrSqrNormDelta(
        float* __restrict__ expectedErrSqrNormDelta, const float damp,
        const HessianVec<Traits, true> delta, const HessianVec<Traits, true> diagBackup,
        HessianVec<Traits, true> g) // Ec, Ea and Eb of hessian
{
    assert(delta.nbCBlocks == diagBackup.nbCBlocks && delta.nbCBlocks == g.nbCBlocks);
    assert(delta.nbABlocks == diagBackup.nbABlocks && delta.nbABlocks == g.nbABlocks);
    assert(delta.nbBBlocks == diagBackup.nbBBlocks && delta.nbBBlocks == g.nbBBlocks);
    uint32_t idxCtaEnd[3];
    makeCtaIdxEnd(idxCtaEnd, delta);
    assert(gridDim.x == idxCtaEnd[2]);
    const uint32_t idxCta = blockIdx.x;
    auto exec = [expectedErrSqrNormDelta, damp](const auto* vecDelta, const auto* vecDiag, const auto* vecG, uint32_t idx, uint32_t idxMax)->void{
        float thrdAcc = 0;
        if (idx < idxMax) {
			const auto elemDelta = vecDelta[idx];
			const auto elemDiag = vecDiag[idx];
			const auto elemG = vecG[idx];
			for (uint32_t i = 0; i < elemDelta.size(); i++)
			{
				thrdAcc += float(elemDelta[i]) * (getDampMu(std::sqrt(float(elemDiag[i])), damp) * float(elemDelta[i]) + elemG[i]);
			}
		}
        warpAtomicAdd<true>(expectedErrSqrNormDelta, thrdAcc);
    };
    if (idxCta < idxCtaEnd[0]) {
        const uint32_t idx = ctaSize * idxCta + threadIdx.x;
        exec(delta.c, diagBackup.c, g.c, idx, delta.nbCBlocks);
    }
    else if (idxCta < idxCtaEnd[1]) {
        const uint32_t idx = ctaSize * (idxCta - idxCtaEnd[0]) + threadIdx.x;
        exec(delta.a, diagBackup.a, g.a, idx, delta.nbABlocks);
    }
    else {
        assert(idxCta < idxCtaEnd[2]);
        const uint32_t idx = ctaSize * (idxCta - idxCtaEnd[1]) + threadIdx.x;
        exec(delta.b, diagBackup.b, g.b, idx, delta.nbBBlocks);
    }
}

template <typename Traits>
cudaError_t launchCudaComputeExpectedErrSqrNormDelta(float* expectedErrSqrNormDelta, float damp,
        const HessianVec<Traits, true>& delta, const HessianVec<Traits, true>& diagBackup,
        const HessianVec<Traits, true>& g, // Ec, Ea and Eb of hessian
        cudaStream_t stream)
{
    checkEarlyReturn(cudaMemsetAsync(expectedErrSqrNormDelta, 0, sizeof(*expectedErrSqrNormDelta), stream));
    uint32_t idxCtaEnd[3];
    makeCtaIdxEnd(idxCtaEnd, delta);
    return launchKernel(kernel_computeExpectedErrSqrNormDelta<Traits>, idxCtaEnd[2], ctaSize, 0, stream, expectedErrSqrNormDelta, damp, delta, diagBackup, g);
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchCudaComputeExpectedErrSqrNormDelta(float* expectedErrSqrNormDelta, float damp,\
            const HessianVec<TRAITS, true>& delta, const HessianVec<TRAITS, true>& diagBackup,\
            const HessianVec<TRAITS, true>& g, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES
}//dampLM

namespace updateModel
{
constexpr uint32_t ctaSize = 128;

template <typename Traits, bool isConst>
__device__ __host__ inline
void makeCtaIdxEnd(uint32_t (&idxCtaEnd)[3], const Model<Traits, isConst>& model){
    dampLM::makeCtaIdxEnd(idxCtaEnd, model.varCameras.nbVar, model.involvedCaps.nbVar, model.varPoints.nbVar);
}

template <typename Traits>
__global__ void kernel_updateModel(const BackupData<Traits, false> backup,
                                   const Model<Traits, false> model, const HessianVec<Traits, true> delta)
{
    uint32_t idxCtaEnd[3];
    makeCtaIdxEnd(idxCtaEnd, model);
    assert(gridDim.x == idxCtaEnd[2]);
    const uint32_t idxCta = blockIdx.x;
    auto exec = [](auto& elemBackup, auto& elem, const auto& elemDelta, uint32_t idx, uint32_t idxEnd)->void{
        assert(idx < idxEnd); unused(idxEnd);
        elemBackup = elem;
        //@fixme: may need negation here for delta
#if 1
        std::decay_t<decltype(elemDelta)> negDelta;
        for (unsigned i = 0; i < elemDelta.size(); i++){
            negDelta[i] = -elemDelta[i];
        }
        elem.update(negDelta.template cast<typename Traits::lpf>());
#else
        elem.update(elemDelta);
#endif
    };
    if (idxCta < idxCtaEnd[0]) {
        const uint32_t idx = ctaSize * idxCta + threadIdx.x;
        if (idx < model.varCameras.nbVar){
            assert(model.cameras[model.varCameras.indices[idx]].varIdx == idx);
			exec(backup.cameras[idx], model.cameras[model.varCameras.indices[idx]].intri,
				delta.c[idx].template cast<typename Traits::lpf>(),
				idx, model.varCameras.nbVar);
        }
    }
    else if (idxCta < idxCtaEnd[1]) {
        const uint32_t idx = ctaSize * (idxCta - idxCtaEnd[0]) + threadIdx.x;
		if (idx < model.involvedCaps.nbVar) {
        	exec(backup.captures[idx], model.captures[model.involvedCaps.indices[idx]], delta.a[idx],
                idx, model.involvedCaps.nbVar);
		}
    }
    else {
        assert(idxCta < idxCtaEnd[2]);
        const uint32_t idx = ctaSize * (idxCta - idxCtaEnd[1]) + threadIdx.x;
        if (idx < model.varPoints.nbVar){
            assert(model.points[model.varPoints.indices[idx]].varIdx == idx);
			exec(backup.points[idx], model.points[model.varPoints.indices[idx]].pt, delta.b[idx],
				idx, model.varPoints.nbVar);
        }
    }
}

template <typename Traits>
cudaError_t launchCudaUpdateModel(const BackupData<Traits, false>& backup, const Model<Traits, false>& model, const HessianVec<Traits, true>& delta, cudaStream_t stream){
    uint32_t idxCtaEnd[3];
    makeCtaIdxEnd(idxCtaEnd, model);
    return launchKernel(kernel_updateModel<Traits>, idxCtaEnd[2], ctaSize, 0, stream, backup, model, delta);
}

#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchCudaUpdateModel(const BackupData<TRAITS, false>& backup, const Model<TRAITS, false>& model, const HessianVec<TRAITS, true>& delta, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template <typename Traits>
__global__ void kernel_revertModel(const Model<Traits, false> model, const BackupData<Traits, true> backup)
{
    uint32_t idxCtaEnd[3];
    makeCtaIdxEnd(idxCtaEnd, model);
    assert(gridDim.x == idxCtaEnd[2]);
    const uint32_t idxCta = blockIdx.x;
    auto exec = [](const auto& elemBackup, auto& elem, uint32_t idx, uint32_t idxMax)->void{
        if (idx >= idxMax) {
            return;
        }
        elem = elemBackup;
    };
    if (idxCta < idxCtaEnd[0]) {
        const uint32_t idx = ctaSize * idxCta + threadIdx.x;
        if (idx < model.varCameras.nbVar){
            assert(model.cameras[model.varCameras.indices[idx]].varIdx == idx);
        }
        exec(backup.cameras[idx], model.cameras[model.varCameras.indices[idx]].intri, idx, model.varCameras.nbVar);
    }
    else if (idxCta < idxCtaEnd[1]) {
        const uint32_t idx = ctaSize * (idxCta - idxCtaEnd[0]) + threadIdx.x;
        exec(backup.captures[idx], model.captures[model.involvedCaps.indices[idx]], idx, model.involvedCaps.nbVar);
    }
    else {
        assert(idxCta < idxCtaEnd[2]);
        const uint32_t idx = ctaSize * (idxCta - idxCtaEnd[1]) + threadIdx.x;
        if (idx < model.varPoints.nbVar){
            assert(model.points[model.varPoints.indices[idx]].varIdx == idx);
        }
        exec(backup.points[idx], model.points[model.varPoints.indices[idx]].pt, idx, model.varPoints.nbVar);
    }
}

template <typename Traits>
cudaError_t launchCudaRevertModel(const Model<Traits, false>& model, const BackupData<Traits, true>& backup, cudaStream_t stream){
    uint32_t idxCtaEnd[3];
    makeCtaIdxEnd(idxCtaEnd, model);
    return launchKernel(kernel_revertModel<Traits>, idxCtaEnd[2], ctaSize, 0, stream, model, backup);
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchCudaRevertModel(const Model<TRAITS, false>& model, const BackupData<TRAITS, true>& backup, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES
} // namespace updateModel
} // namespace rba
