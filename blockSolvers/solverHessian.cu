//
// Created by yao on 8/12/18.
//
#include "../cuda_hint.cuh"
#include "solveHessian.h"
#include "../kernel.h"
#include <cooperative_groups.h>
#include "../derivative.h"
#include <cassert>
#include <boost/preprocessor/seq/for_each.hpp>
namespace cg = cooperative_groups;

namespace rba
{
namespace solve_deltaB {
template <typename Traits>
__global__ void kernel_computeSTDeltaC(
        typename HessianBase<Traits>::EbBlock *__restrict__ acc,
        const CSR<const typename HessianBase<Traits>::SBlock> hessianS,
        const typename HessianBase<Traits>::EcBlock *__restrict__ deltaCBlocks) {
    RBA_IMPORT_TRAITS(Traits);
    constexpr unsigned ctaSize = 128;
    assert(blockDim.x == ctaSize);
    const unsigned &idxM = blockIdx.y;
    assert(gridDim.y == hessianS.nbRows);
    __shared__ struct {
        uint32_t start;
        uint32_t end;
        kmat<lpf, HessianBase<Traits>::EcBlock::rows()> deltaC;
    } task;
    if (threadIdx.x == 0) {
        const unsigned gridSizeX = blockDim.x * gridDim.x;
        const uint32_t rowBeg = hessianS.getRowBegin(idxM);
        const uint32_t rowEnd = hessianS.getRowEnd(idxM);
        const uint32_t rowSize = rowEnd - rowBeg;
        const unsigned targetItems = divUp(rowSize, gridSizeX) * blockDim.x;
        const uint32_t start = rowBeg + targetItems * blockIdx.x;
        task.start = start;
        task.end = std::min(start + targetItems, rowEnd);
    }
    assert(task.deltaC.size() < ctaSize);
    if (threadIdx.x < task.deltaC.size())
        task.deltaC[threadIdx.x] = static_cast<lpf>(deltaCBlocks[idxM][threadIdx.x]);
    __syncthreads();
    if (task.start >= task.end)
        return;

    IdxVBlock idxV[2]{};
    typename HessianBase<Traits>::SBlock sBlock[2];
    auto rotate = [&]() {
        idxV[0] = idxV[1];
        sBlock[0] = sBlock[1];
    };
    auto prefetch = [&](uint32_t idx) {
        if (idx < task.end) {
            idxV[1] = hessianS.idxCol[idx];
            sBlock[1] = hessianS.data[idx];
        }
#ifndef NDEBUG
        else {
            idxV[1] = ~0u;
            sBlock[1].assignScalar(NAN);
        }
#endif
    };

    prefetch(threadIdx.x);
    for (uint32_t idx = task.start + threadIdx.x; idx < task.end; idx += ctaSize) {
        rotate();
        prefetch(idx + ctaSize);
        const typename HessianBase<Traits>::EbBlock STDeltaC =
                idx < task.end ? sBlock[0].transpose() * task.deltaC : HessianBase<Traits>::EbBlock::zeros();
        assert(STDeltaC.size() == acc[0].size());
        for (int i = 0; i < STDeltaC.size(); i++) {
            //@fixme: transpose first
            atomicAdd(acc[idxV[0]].data() + i, -STDeltaC[i]);//@info: note here we use negative
        }
    }
}

template <typename Traits>
cudaError_t launchCudaComputeSTDeltaC(
        typename HessianBase<Traits>::EbBlock *acc, const CSR<const typename HessianBase<Traits>::SBlock> &hessianS,
        const typename HessianBase<Traits>::EcBlock *deltaCBlocks, cudaStream_t stream) {
    if (hessianS.nbRows == 0)
        return cudaSuccess;
    const uint32_t ctaSize = 128;
    const dim3 dimGrid = {divUp(128u, hessianS.nbRows), hessianS.nbRows};
    kernel_computeSTDeltaC<Traits><<<dimGrid, ctaSize, 0, stream>>> (acc, hessianS, deltaCBlocks);
    return cudaGetLastError();
}

//refer to computeSchurDiagU in computeSchur.cu
template <typename Traits>
__global__ void kernel_computeWTDeltaA(
        typename HessianBase<Traits>::EbBlock *__restrict__ acc,
        const Model<Traits, true> model,
        const VecVec<const uint16_t> sparseIdxW2idxLocalOb, // list of local ob index for variable points. nbRows = nbVarCaps
        const typename HessianBase<Traits>::EaBlock *__restrict__ deltaABlocks,
        const CSR<const typename HessianBase<Traits>::WBlock> hessianW // only for ref-check
) {
    RBA_IMPORT_TRAITS(Traits);
    constexpr unsigned ctaSize = 128;
    assert(blockDim.x == ctaSize);
    constexpr unsigned grpSize = warp_size;
    constexpr unsigned grpsPerCta = ctaSize / grpSize;
    const auto g = cg::tiled_partition<grpSize>(cg::this_thread_block());
    assert(blockDim.x == ctaSize && blockDim.y == 1 && blockDim.z == 1);
    assert(gridDim.x * grpsPerCta >= model.involvedCaps.nbVar && gridDim.y == 1 && gridDim.z == 1);
    const unsigned idxU = blockIdx.x * grpsPerCta + threadIdx.x / grpSize;
    if (idxU >= sparseIdxW2idxLocalOb.getNbRows())
        return;

    __shared__ struct {
        IdxUBlock idxU;
        uint32_t idxCap;
        const uint16_t *sparseIdxW2idxLocalOb;
        uint32_t nbVarObs;
        Capture capture;
        typename Traits::CamIntr camera;
        const CapOb<lpf> *obs;
        uint32_t nbObs;
        kmat<bool, 3> fixedCapLoc;
        typename HessianBase<Traits>::EaBlock deltaA;
    } tasks[grpsPerCta];
#if USE_TRANSPOSED_REDUCTION
#endif
    {
        auto &task = tasks[grpSize == ctaSize ? 0 : threadIdx.x / grpSize];
        if (g.thread_rank() == 0) {
            task.sparseIdxW2idxLocalOb = sparseIdxW2idxLocalOb[idxU];
            task.nbVarObs = sparseIdxW2idxLocalOb.getRowSize(idxU);
            assert(sparseIdxW2idxLocalOb.getNbRows() == model.involvedCaps.nbVar);
            assert(task.nbVarObs == hessianW.getRowLength(idxU));
            assert(idxU < model.involvedCaps.nbVar);
            const auto idxCap = model.involvedCaps.indices[idxU];
            task.idxCap = idxCap;
            task.capture = model.captures[idxCap];
            task.obs = model.capObs[idxCap];
            task.nbObs = model.capObs.getRowSize(idxCap);
        }
        g.sync();
        if (g.thread_rank() < 3) {
            task.fixedCapLoc[g.thread_rank()] = (model.gnssCtrl != nullptr) && model.gnssCtrl[task.idxCap].isHard(g.thread_rank());
        }
        if (isGroupModel<Traits>()) {
            for (unsigned i = 0; i < divUp(sizeof(task.camera), sizeof(Traits::lpf)); i++) {
                const auto idx = grpSize * i + g.thread_rank();
                assert(sizeof(Traits::CamIntr) % sizeof(Traits::lpf) == 0);
                if (idx < sizeof(task.camera) / sizeof(Traits::lpf)) {
                    reinterpret_cast<lpf *>(&task.camera)[idx] =
                            reinterpret_cast<const lpf *>(&model.cameras[task.capture.intrIdx].intri)[idx];
                }
            }
        }
        assert(task.deltaA.size() <= grpSize);
        if (g.thread_rank() < task.deltaA.size()) {
            task.deltaA.data()[g.thread_rank()] = deltaABlocks[idxU].data()[g.thread_rank()];
        }
    }
    g.sync();
    const auto &task = tasks[grpSize == ctaSize ? 0 : threadIdx.x / grpSize];
    uint16_t idxLocalOb[4];
    CapOb<lpf> obs[3];
    typename ModelBase<Traits>::PointWVarIdx points[2];

#pragma unroll
    for (uint32_t i = 1; i < 4; i++) {
        const uint32_t idxPre = g.size() * (i - 1) + g.thread_rank();
        const bool isInRange = idxPre < task.nbVarObs;
        if (i < 4)
            idxLocalOb[i] = isInRange ? task.sparseIdxW2idxLocalOb[idxPre] : uint16_t(0xFFFFu);
        if (i < 3) {
            if (isInRange) {
                assert(idxLocalOb[i] < task.nbObs);
                obs[i] = task.obs[idxLocalOb[i]];
            }
#ifndef NDEBUG
            else
                obs[i].ptIdx = ~0u;
#endif
        }
        if (i < 2) {
            if (isInRange) {
                points[i] = model.points[obs[i].ptIdx];
            }
#ifndef NDEBUG
            else {
                points[i] = {{NAN, NAN, NAN}, ModelBase<Traits>::varIdxFixed};
            }
#endif
        }
    }

    for (uint32_t n = 0; n < divUp(task.nbVarObs, g.size()); n++) {
        const uint32_t idx = g.size() * n + g.thread_rank();//sparseIdxW
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint32_t idxPre = idx + g.size() * i;
            const bool isInRange = idxPre < task.nbVarObs;
            // rotate
            if (i < 3)
                idxLocalOb[i] = idxLocalOb[i + 1];
            if (i < 2)
                obs[i] = obs[i + 1];
            if (i < 1) {
                points[i] = points[i + 1];
            }
            // prefetch
            if (i == 3)
                idxLocalOb[i] = isInRange ? task.sparseIdxW2idxLocalOb[idxPre] : uint16_t(0xFFFFu);
            if (i == 2) {
                if (isInRange) {
                    assert(idxLocalOb[i] < task.nbObs);
                    obs[i] = task.obs[idxLocalOb[i]];
                } else
                    obs[i].ptIdx = ~0u;
            }
            if (i == 1) {
                if (isInRange) {
                    points[i] = model.points[obs[i].ptIdx];
                }
#ifndef NDEBUG
                else {
                    points[i] = {{NAN, NAN, NAN}, ModelBase<Traits>::varIdxFixed};
                }
#endif
            }
        }
        const auto errDeriv = computeErrorDerivative<Traits>(task.camera, task.capture,
                                                     toKmat(points[0].pt.position), {obs[0].position},
                                                     task.fixedCapLoc);
        const lpf weightedOmega = robustify<lpf>(errDeriv.error, obs[0].omega, obs[0].huber);
#ifndef NDEBUG
        // ref-check for W
        if (idx < task.nbVarObs && hessianW.data != nullptr) {
            const typename HessianBase<Traits>::WBlock W = errDeriv.jacobian.capture.transpose() * (weightedOmega * errDeriv.jacobian.pt);
            lpf absMaxW = 0.f;
            for (int i = 0; i < W.size(); i++) {
                absMaxW = std::max(absMaxW, std::abs(W.data()[i]));
            }
            for (int i = 0; i < W.size(); i++) {
                assert(std::abs(W.data()[i] - hessianW[idxU][idx].data()[i]) <
                       0.01f * std::max(Traits::lpf(1.f), absMaxW));
            }
        }
#endif
        typename HessianBase<Traits>::EbBlock WTDeltaA =
                weightedOmega * (errDeriv.jacobian.pt.transpose() * (errDeriv.jacobian.capture * task.deltaA.template cast<lpf>()));
        if (idx >= task.nbVarObs) {
            WTDeltaA = HessianBase<Traits>::EbBlock::zeros();
        }
        if (idx < task.nbVarObs) {
            assert(WTDeltaA.size() == acc[0].size());
            for (int i = 0; i < WTDeltaA.size(); i++) {
                //@fixme: transpose first
                atomicAdd(acc[points[0].varIdx].data() + i, -WTDeltaA.data()[i]);//@info: note here we use negative
            }
        }
    }
}

template <typename Traits>
cudaError_t launchCudaComputeWTDeltaA(
        typename HessianBase<Traits>::EbBlock *acc, const Model<Traits, true> &model,
        const VecVec<const uint16_t> &sparseIdxW2idxLocalOb, // list of local ob index for variable points. nbRows = nbVarCaps
        const typename HessianBase<Traits>::EaBlock *deltaABlocks,
        const CSR<const typename HessianBase<Traits>::WBlock> &hessianW,
        cudaStream_t stream) {
    if (model.involvedCaps.nbVar == 0)
        return cudaSuccess;
    const uint32_t ctaSize = 128;
    const uint32_t dimGrid = divUp(model.involvedCaps.nbVar, ctaSize / warp_size);
    kernel_computeWTDeltaA << < dimGrid, ctaSize, 0, stream >> >
                                                     (acc, model, sparseIdxW2idxLocalOb, deltaABlocks, hessianW);
    return cudaGetLastError();
}

template<typename Traits, typename SymBlockType = typename HessianBase<Traits>::VSymBlock, typename VecType = typename HessianBase<Traits>::EbBlock>
__global__ void kernel_computeMVInplace(
        VecType *__restrict__ vectors,
        const SymBlockType *__restrict__ symMatrices,
        uint32_t nbBlocks) {
    static_assert(std::is_same<typename SymBlockType::ValType, typename VecType::ValType>::value, "fatal error");
    const unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nbBlocks)
        return;
    const SymBlockType m = symMatrices[idx];
    const VecType a = vectors[idx];
    const VecType vec = m.toKMat() * a;
    vectors[idx] = vec;
}

template <typename Traits>
cudaError launchCudaComputeMVInplace(typename HessianBase<Traits>::EbBlock* vectors, const typename HessianBase<Traits>::VSymBlock* symMatrices,
        uint32_t nbBlocks, cudaStream_t stream)
{
    if (nbBlocks == 0)
        return cudaSuccess;
    const uint32_t ctaSize = 128;
    const uint32_t nbCta = divUp(nbBlocks, ctaSize);
    kernel_computeMVInplace<Traits><<<nbCta, ctaSize, 0, stream>>>(vectors, symMatrices, nbBlocks);
    return cudaGetLastError();
}

#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchCudaComputeSTDeltaC<TRAITS>(\
        typename HessianBase<TRAITS>::EbBlock *acc, const CSR<const typename HessianBase<TRAITS>::SBlock> &hessianS,\
        const typename HessianBase<TRAITS>::EcBlock *deltaCBlocks, cudaStream_t stream);\
    template cudaError launchCudaComputeMVInplace<TRAITS>(typename HessianBase<TRAITS>::EbBlock* vectors,\
        const typename HessianBase<TRAITS>::VSymBlock* symMatrices,\
        uint32_t nbBlocks, cudaStream_t stream);\
    template cudaError_t launchCudaComputeWTDeltaA<TRAITS>(\
        typename HessianBase<TRAITS>::EbBlock *acc, const Model<TRAITS, true> &model,\
        const VecVec<const uint16_t> &sparseIdxW2idxLocalOb, \
        const typename HessianBase<TRAITS>::EaBlock *deltaABlocks,\
        const CSR<const typename HessianBase<TRAITS>::WBlock> &hessianW,\
        cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

} // namespace hessianB
} // namespace rba
