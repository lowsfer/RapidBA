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
// Created by yao on 11/11/18.
//
#include "cuda_hint.cuh"
#include "computeSchur.h"
#include <cuda_runtime_api.h>
#include "csr.h"
#include <cooperative_groups.h>
#include "utils_kernel.h"
#include "kernel.h"
#include "derivative.h"
#include "computeHessian.h"
#include "transpose.cuh"
#include <boost/preprocessor/seq/for_each.hpp>
#include <cstdio>

#define USE_TRANSPOSED_REDUCTION 1

namespace cg = cooperative_groups;

namespace rba {


namespace SchurUpperMComputer {
template <typename Traits, unsigned ctaSize = 128>
__global__ void compute(
        const Hessian<Traits, true> hessian, const typename HessianBase<Traits>::VSymBlock *__restrict__ invV,
        const Schur<Traits, false> schur, const PreComp preComp) {
    using lpf = typename Traits::lpf;
    using PairSS = PreComp::PairSS;
    using SBlock = typename HessianBase<Traits>::SBlock;
    using VSymBlock = typename HessianBase<Traits>::VSymBlock;
    assert(blockDim.x == ctaSize && blockDim.y == 1 && blockDim.z == 1);
    assert(gridDim.z == 1);
    assert(preComp.pairSS.getNbRows() == gridDim.y);
    assert(schur.nbVarIntri == hessian.nbVarIntri && schur.nbVarIntri == hessian.S.nbRows);
    assert(schur.upperM.nbRows == schur.nbVarIntri);
    const unsigned idxItem = blockIdx.y;
    __shared__ IdxMBlock idxA;
    __shared__ IdxMBlock idxB;
    __shared__ const PairSS *ctaPairs;
    __shared__ uint32_t nbCtaPairs;
    __shared__ uint32_t outerLoop;
    __shared__ const SBlock *dataA;
    __shared__ const SBlock *dataB;
    __shared__ const IdxVBlock *idxColA;
    __shared__ const IdxVBlock *idxColB;
    __shared__ typename Schur<Traits, false>::MBlock smemM;
    //@fixme: set innerLoop to 1 or 2 to reduce code size and register usage, but I don't know the performance impact, yet. with 2 or 4, compiler should be able to do some prefetch
    constexpr int innerLoop = 1;
    if (threadIdx.x == 0) {
        idxA = preComp.rowTableUpperM[idxItem];
        idxB = schur.upperM.idxCol[idxItem];
        assert(idxA < idxB && idxB < schur.nbVarIntri);
        const auto rowSize = preComp.pairSS.getRowSize(idxItem);
        const uint32_t targetNbCtaPairs = (ctaSize * innerLoop) * divUp(rowSize, ctaSize * innerLoop * gridDim.x);
        if (blockIdx.x < divUp(rowSize, targetNbCtaPairs)) {
            ctaPairs = &preComp.pairSS[idxItem][targetNbCtaPairs * blockIdx.x];
            nbCtaPairs = std::min(targetNbCtaPairs, rowSize - targetNbCtaPairs * blockIdx.x);
        } else {
            ctaPairs = nullptr;
            nbCtaPairs = 0;
        }
        outerLoop = divUp(nbCtaPairs, innerLoop * ctaSize);
        dataA = &hessian.S.data[hessian.S.getRowBegin(idxA)];
        dataB = &hessian.S.data[hessian.S.getRowBegin(idxB)];
        idxColA = &hessian.S.idxCol[hessian.S.getRowBegin(idxA)];
        idxColB = &hessian.S.idxCol[hessian.S.getRowBegin(idxB)];
    }
    for (unsigned i = threadIdx.x; i < smemM.size(); i += blockDim.x) {
        smemM.data()[i] = 0;
    }
    __syncthreads();
    if (nbCtaPairs == 0)
        return;

    for (uint32_t n = 0; n < outerLoop; n++) {
        const unsigned idxBase = innerLoop * ctaSize * n;
        if (idxBase + (threadIdx.x & ~(warp_size - 1)) >= nbCtaPairs) {
            assert((__syncwarp(), __activemask() == ~0u));
            break;
        }
        kmat<lpf, smemM.rows(), smemM.cols()> thrdM = kmat<lpf, smemM.rows(), smemM.cols()>::zeros();
#pragma unroll
        for (uint32_t i = 0; i < innerLoop; i++) {
            const uint32_t idxPair = idxBase + ctaSize * i + threadIdx.x;
            if (idxPair >= nbCtaPairs)
                break;
            const PairSS pairAB = ctaPairs[idxPair];
            assert(idxColA[pairAB.a] == idxColB[pairAB.b]); unused(idxColB);
            const SBlock blockA = dataA[pairAB.a];
            const SBlock blockB = dataB[pairAB.b];
            const VSymBlock invVBlock = invV[idxColA[pairAB.a]];
            thrdM = thrdM + blockA * invVBlock.toKMat() * blockB.transpose();
        }
        // @fixme: consider transpose first
        for (unsigned j = 0; j < thrdM.size(); j++) {
            static_assert(thrdM.size() == smemM.size(), "fatal error");
            assert((__syncwarp(), __activemask() == ~0u));
            warpAtomicAdd<true>(&smemM.data()[j], thrdM.data()[j]);
        }
    }
    __syncthreads();
    assert(schur.upperM.getRowBegin(idxA) <= idxItem && idxItem < schur.upperM.getRowEnd(idxB));
    assert(schur.upperM.idxCol[idxItem] == idxB);
    for (unsigned i = threadIdx.x; i < smemM.size(); i += ctaSize) {
        atomicAdd(&schur.upperM.data[idxItem].data()[i], -smemM.data()[i]);
    }
}
}//namespace SchurUpperMComputer

template <typename Traits>
cudaError_t launchCudaComputeSchurUpperM(
        const Hessian<Traits, true> hessian, const typename HessianBase<Traits>::VSymBlock *__restrict__ invV,
        const Schur<Traits, false> schur, const SchurUpperMComputer::PreComp preComp, cudaStream_t stream) {
    if (preComp.pairSS.getNbRows() == 0)
        return cudaGetLastError();
    constexpr unsigned ctaSize = 128;
    const dim3 dimBlock(ctaSize);
    const dim3 dimGrid(divUp(128u, preComp.pairSS.getNbRows()), preComp.pairSS.getNbRows());
    return launchKernel(SchurUpperMComputer::compute<Traits, ctaSize>, dimGrid, dimBlock, 0, stream, hessian, invV, schur, preComp);
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchCudaComputeSchurUpperM(\
            const Hessian<TRAITS, true> hessian, const typename HessianBase<TRAITS>::VSymBlock *__restrict__ invV,\
            const Schur<TRAITS, false> schur, const SchurUpperMComputer::PreComp preComp, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template<typename Traits, unsigned ctaSize = 128>
__global__ void kernel_computeSchurDiagM(const Hessian<Traits, true> hessian, const typename HessianBase<Traits>::VSymBlock *__restrict__ invV,
                                         const Schur<Traits, false> schur) {
    RBA_IMPORT_TRAITS(Traits);
    assert(blockDim.x == ctaSize && blockDim.y == 1 && blockDim.z == 1);
    assert(gridDim.z == 1);
    assert(schur.nbVarIntri == gridDim.y && schur.nbVarIntri == hessian.nbVarIntri &&
           schur.nbVarIntri == hessian.S.nbRows);
    //@fixme: set innerLoop to 1 or 2 to reduce code size and register usage, but I don't know the performance impact, yet. With 2 or 4, compiler should be able to do some prefetch
    constexpr int innerLoop = 1;
    const unsigned idxM = blockIdx.y;
    __shared__ uint32_t nbSBlocks;
    __shared__ const typename Hessian<Traits, true>::SBlock *sBlocks;
    __shared__ const IdxMBlock *idxCol;
    __shared__ uint32_t outerLoop;
    __shared__ typename Schur<Traits, false>::MSymBlock smemM;
    __shared__ typename Schur<Traits, false>::EcBlock smemEc;
    if (threadIdx.x == 0) {
        const auto rowBeg = hessian.S.getRowBegin(idxM);
        const auto rowSize = hessian.S.getRowEnd(idxM) - rowBeg;
        const uint32_t targetNbSBlocks = (ctaSize * innerLoop) * divUp(rowSize, ctaSize * innerLoop * gridDim.x);
        if (blockIdx.x < divUp(rowSize, targetNbSBlocks)) {
            sBlocks = &hessian.S.data[rowBeg + targetNbSBlocks * blockIdx.x];
            idxCol = &hessian.S.idxCol[rowBeg + targetNbSBlocks * blockIdx.x];
            nbSBlocks = std::min(targetNbSBlocks, rowSize - targetNbSBlocks * blockIdx.x);
        } else {
            sBlocks = nullptr;
            idxCol = nullptr;
            nbSBlocks = 0;
        }
        outerLoop = divUp(nbSBlocks, innerLoop * ctaSize);
    }
    for (int i = threadIdx.x; i < smemM.size(); i += blockDim.x) {
        smemM.data()[i] = 0;
    }
    for (int i = threadIdx.x; i < smemEc.size(); i += blockDim.x) {
        smemEc.data()[i] = 0;
    }
    __syncthreads();

    for (int n = 0; n < outerLoop; n++) {
        const unsigned idxBase = innerLoop * ctaSize * n;
        if (idxBase + (threadIdx.x & ~(warp_size - 1)) >= nbSBlocks) {
            assert((__syncwarp(), __activemask() == ~0u));
            break;
        }
        symkmat<lpf, HessianBase<Traits>::MSymBlock::rows()> thrdM = kmat<lpf, HessianBase<Traits>::MSymBlock::rows(), HessianBase<Traits>::MSymBlock::cols()>::zeros();
        kmat<lpf, smemEc.size()> thrdEc = kmat<lpf, smemEc.size()>::zeros();
        for (unsigned i = 0; i < innerLoop; i++) {
            const unsigned idx = idxBase + ctaSize * i + threadIdx.x;
            if (idx >= nbSBlocks)
                break;
            const typename HessianBase<Traits>::SBlock blockS = sBlocks[idx];
            const auto idxV = idxCol[idx];
            const typename HessianBase<Traits>::VSymBlock invBlockV = invV[idxV];
            const typename HessianBase<Traits>::EbBlock Eb = hessian.Eb[idxV];
            //thrdM = thrdM + blockS * invBlockV.toKMat() * blockS.transpose(); //@fixme: this version will take 32 more registers, why?
            thrdM = thrdM.toKMat() + blockS * invBlockV.toKMat() * blockS.transpose();
            thrdEc = thrdEc + blockS * invBlockV.toKMat() * Eb;
        }
        // @fixme: consider transpose first
        for (unsigned i = 0; i < thrdM.size(); i++) {
            static_assert(thrdM.size() == smemM.size(), "fatal error");
            assert((__syncwarp(), __activemask() == ~0u));
            warpAtomicAdd<true>(&smemM.data()[i], thrdM.data()[i]);
        }
        for (unsigned i = 0; i < thrdEc.size(); i++) {
            static_assert(thrdEc.size() == smemEc.size(), "fatal error");
            assert((__syncwarp(), __activemask() == ~0u));
            warpAtomicAdd<true>(&smemEc.data()[i], thrdEc.data()[i]);
        }
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < smemM.size(); i += ctaSize) {
        atomicAdd(&schur.diagM[idxM].data()[i], -smemM.data()[i]);//@info: note here we use negative smemM
    }
    for (unsigned i = threadIdx.x; i < smemEc.size(); i += ctaSize) {
        atomicAdd(&schur.Ec[idxM].data()[i], -smemEc.data()[i]);//@info: note here we use negative smemEc
    }
}

template <typename Traits>
cudaError_t launchCudaComputeSchurDiagM(const Hessian<Traits, true> hessian, const typename HessianBase<Traits>::VSymBlock *__restrict__ invV,
                                        const Schur<Traits, false> schur, cudaStream_t stream) {
    if (!isGroupModel<Traits>() || schur.nbVarIntri == 0){
        return cudaGetLastError();
    }
    constexpr unsigned ctaSize = 128;
    const dim3 dimBlock(ctaSize);
    assert(hessian.nbVarIntri == schur.nbVarIntri);
    // dimGrid.x = divUp(128u, schur.nbVarIntri) means when we have enough diagM, we use less CTAs to compute one block
    const dim3 dimGrid(divUp(128u, schur.nbVarIntri), schur.nbVarIntri);
    return launchKernel(kernel_computeSchurDiagM<Traits>, dimGrid, dimBlock, 0, stream, hessian, invV, schur);
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchCudaComputeSchurDiagM(const Hessian<TRAITS, true> hessian, const typename HessianBase<TRAITS>::VSymBlock *__restrict__ invV,\
                                            const Schur<TRAITS, false> schur, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

// @fixme: Try use existing Hessian W blocks instead to re-compute to see if performance is better.
namespace SchurUpperUComputer {
template<typename Traits, uint32_t ctaSize = 128>
__global__ void compute(
        const Model<Traits, true> model,
        const VecVec<const ObPair> taskList, // nbRows = nbUpperUItems
        const Hessian<Traits, true> hessian, //hessian is not needed. Put it here just for validation with assertions
        const typename HessianBase<Traits>::VSymBlock *__restrict__ invVList,
        const Schur<Traits, false> schur,
        const IdxUBlock *__restrict__ rowTableUpperU // same structure as schur.upperU.idxCols
) {
    RBA_IMPORT_TRAITS(Traits);
    constexpr unsigned grpSize = warp_size;
    constexpr unsigned grpsPerCta = ctaSize / grpSize;
    const auto g = cg::tiled_partition<grpSize>(cg::this_thread_block());

    assert(blockDim.x == ctaSize && blockDim.y == 1 && blockDim.z == 1);
    assert(gridDim.x * grpsPerCta >= schur.upperU.nbElems() && gridDim.y == 1 && gridDim.z == 1);
    const unsigned idxItem = blockIdx.x * grpsPerCta + threadIdx.x / grpSize;
    if (idxItem >= taskList.getNbRows())
        return;
#if USE_TRANSPOSED_REDUCTION
    using Reducer = GroupReducer<HessianBase<Traits>::UBlock::size(), grpSize>;
    __shared__ typename Reducer::Buffer transBuffers[grpsPerCta];
#endif
    __shared__ struct {
        uint32_t idxItem;
        IdxUBlock idxA;
        IdxUBlock idxB;
        const ObPair *pairs;
        uint32_t nbPairs;
        struct {
            uint32_t idxCap;
            Capture capture;
            CamIntr camera;
            const CapOb<lpf> *obs;
            uint32_t nbObs;
            kmat<bool, 3> fixedCapLoc;
        } camA, camB;
        mutable typename HessianBase<Traits>::UBlock smemU;
    } tasks[grpsPerCta];
    {
        auto &task = tasks[grpSize == ctaSize ? 0 : threadIdx.x / grpSize];
        if (g.thread_rank() == 0) {
            task.idxItem = idxItem;
            task.idxA = rowTableUpperU[idxItem];
            task.idxB = schur.upperU.idxCol[idxItem];
            task.pairs = taskList[idxItem];
            task.nbPairs = taskList.getRowSize(idxItem);
            assert(task.idxA < model.involvedCaps.nbVar && task.idxB < model.involvedCaps.nbVar);
            const auto idxCapA = model.involvedCaps.indices[task.idxA];
            task.camA.idxCap = idxCapA;
            const auto idxCapB = model.involvedCaps.indices[task.idxB];
            task.camB.idxCap = idxCapB;
            task.camA.capture = model.captures[idxCapA];
            task.camB.capture = model.captures[idxCapB];
            task.camA.obs = model.capObs[idxCapA];
            task.camB.obs = model.capObs[idxCapB];
            task.camA.nbObs = model.capObs.getRowSize(idxCapA);
            task.camB.nbObs = model.capObs.getRowSize(idxCapB);
        }
        g.sync();
        if (g.thread_rank() < 3) {
            const auto i = g.thread_rank();
            task.camA.fixedCapLoc[i] = (model.gnssCtrl != nullptr) && model.gnssCtrl[task.camA.idxCap].isHard(i);
            task.camB.fixedCapLoc[i] = (model.gnssCtrl != nullptr) && model.gnssCtrl[task.camB.idxCap].isHard(i);
        }
        if (isGroupModel<Traits>()) {
            for (unsigned i = 0; i < divUp(sizeof(task.camA.camera), sizeof(Traits::lpf)); i++) {
                const auto idx = grpSize * i + g.thread_rank();
                assert(sizeof(CamIntr) % sizeof(Traits::lpf) == 0);
                if (idx < sizeof(task.camA.camera) / sizeof(Traits::lpf)) {
                    reinterpret_cast<lpf *>(&task.camA.camera)[idx] =
                            reinterpret_cast<const lpf *>(&model.cameras[task.camA.capture.intrIdx].intri)[idx];
                    reinterpret_cast<lpf *>(&task.camB.camera)[idx] =
                            reinterpret_cast<const lpf *>(&model.cameras[task.camB.capture.intrIdx].intri)[idx];
                }
            }
        }
        for (unsigned i = 0; i < divUp(task.smemU.size(), grpSize); i++) {
            const auto idx = grpSize * i + g.thread_rank();
            if (idx < task.smemU.size())
                task.smemU.data()[idx] = 0;
        }
    }
    g.sync();
    const auto &task = tasks[grpSize == ctaSize ? 0 : threadIdx.x / grpSize];
    ObPair pairs[5];
    struct {
        CapOb<lpf> obA;
        CapOb<lpf> obB;
    } obs[4];
    typename ModelBase<Traits>::PointWVarIdx points[3];
    typename HessianBase<Traits>::VSymBlock invVBlocks[2];

#pragma unroll
    for (int i = 1; i < 5; i++) {
        const uint32_t idxPre = g.size() * (i - 1) + g.thread_rank();
        const bool isInRange = idxPre < task.nbPairs;
        if (i < 5)
            pairs[i] = isInRange ? task.pairs[idxPre] : ObPair{0xFFFFu, 0xFFFFu};
        if (i < 4) {
            if (isInRange) {
                assert(pairs[i].idxLocalObA < task.camA.nbObs && pairs[i].idxLocalObB < task.camB.nbObs);
                obs[i] = {task.camA.obs[pairs[i].idxLocalObA], task.camB.obs[pairs[i].idxLocalObB]};
            } else
                obs[i].obA.ptIdx = obs[i].obB.ptIdx = ~0u;
        }
        if (i < 3) {
            if (isInRange) {
                assert(obs[i].obA.ptIdx == obs[i].obB.ptIdx && obs[i].obA.ptIdx < model.nbPts);
                points[i] = model.points[obs[i].obA.ptIdx];
            }
        }
        if (i < 2) {
            if (isInRange) {
                assert(points[i].varIdx < model.varPoints.nbVar);
                invVBlocks[i] = invVList[points[i].varIdx];
            }
        }
    }
    for (uint32_t n = 0; n < divUp(task.nbPairs, g.size()); n++) {
        const uint32_t idx = g.size() * n + g.thread_rank();
		typename Schur<Traits, false>::UBlock U;
		if (idx < task.nbPairs){		
			#pragma unroll
			for (int i = 0; i < 5; i++) {
				const uint32_t idxPre = idx + g.size() * i;
				const bool isInRange = idxPre < task.nbPairs;
				// rotate
				if (i < 4)
					pairs[i] = pairs[i + 1];
				if (i < 3)
					obs[i] = obs[i + 1];
				if (i < 2)
					points[i] = points[i + 1];
				if (i < 1)
					invVBlocks[i] = invVBlocks[i + 1];
				// prefetch
				if (i == 4)
					pairs[i] = isInRange ? task.pairs[idxPre] : ObPair{0xFFFFu, 0xFFFFu};
				if (i == 3) {
					if (isInRange) {
						assert(pairs[i].idxLocalObA < task.camA.nbObs && pairs[i].idxLocalObB < task.camB.nbObs);
						obs[i] = {task.camA.obs[pairs[i].idxLocalObA], task.camB.obs[pairs[i].idxLocalObB]};
					} else
						obs[i].obA.ptIdx = obs[i].obB.ptIdx = ~0u;
				}
				if (i == 2) {
					if (isInRange) {
						assert(obs[i].obA.ptIdx == obs[i].obB.ptIdx && obs[i].obA.ptIdx < model.nbPts);
						points[i] = model.points[obs[i].obA.ptIdx];
					}
				}
				if (i == 1) {
					if (isInRange) {
						assert(points[i].varIdx < model.varPoints.nbVar);
						invVBlocks[i] = invVList[points[i].varIdx];
					}
				}
			}
			const auto errDerivA = computeErrorDerivative<Traits>(task.camA.camera, task.camA.capture,
				toKmat(points[0].pt.position), {obs[0].obA.position},
				task.camA.fixedCapLoc);
			const lpf weightedOmegaA = robustify<lpf>(errDerivA.error, obs[0].obA.omega, obs[0].obA.huber);
			//@fixme: finish ref-check
//#ifndef NDEBUG
//            // ref-check for W
//            if (idxXXX < task.nbVarObs && hessian.W.data != nullptr) {
//                const typename HessianBase<Traits>::WBlock W = errDerivA.jacobian.capture.transpose() * (weightedOmegaA * errDerivA.jacobian.pt);
//                for (int i = 0; i < W.size(); i++)
//                    assert(std::abs(W.data()[i] - hessian.W[task.idxA][...].data()[i]) < 1E-2f);
//            }
//#endif
			const typename Traits::Intrinsics intrinsicsB = getIntrinsics(task.camB.camera, task.camB.capture);
			const Pose<lpf>& poseB = task.camB.capture.getPose();
			const auto errDerivB = computeErrorDerivative<Traits>(task.camB.camera, task.camB.capture,
				toKmat(points[0].pt.position), {obs[0].obB.position},
				task.camB.fixedCapLoc);
			const lpf weightedOmegaB = robustify<lpf>(errDerivB.error, obs[0].obB.omega, obs[0].obB.huber);
//#ifndef NDEBUG
//        // ref-check for W
//            if (idxXXX < task.nbVarObs && hessian.W.data != nullptr) {
//                const typename HessianBase<Traits>::WBlock W = errDerivB.jacobian.capture.transpose() * (weightedOmegaB * errDerivB.jacobian.pt);
//                for (int i = 0; i < W.size(); i++)
//                    assert(std::abs(W.data()[i] - hessian.W[task.idxB][...].data()[i]) < 1E-2f);
//            }
//#endif
        	U = (weightedOmegaA * weightedOmegaB * errDerivA.jacobian.capture.transpose() *
                                 (errDerivA.jacobian.pt * invVBlocks[0].toKMat() * errDerivB.jacobian.pt.transpose()) *
                                 errDerivB.jacobian.capture).template cast<hpf>();
		}
		else {
            U = Schur<Traits, false>::UBlock::zeros();
		}

        assert(U.size() == task.smemU.size());
#if USE_TRANSPOSED_REDUCTION
        constexpr uint32_t sizeU = decltype(U)::size();
        typename HessianBase<Traits>::UBlock::ValType grpRedU[divUp(sizeU, grpSize)];
        Reducer::reduce(g, transBuffers[grpSize == ctaSize ? 0 : threadIdx.x / grpSize], grpRedU, U.data());
        for (int i = 0; i < divUp(sizeU, grpSize); i++){
            if (grpSize*(i+1) <= sizeU || g.thread_rank() < sizeU % grpSize)
                atomicAdd(&task.smemU.data()[grpSize*i+g.thread_rank()], grpRedU[i]);
        }
#else
        for (int i = 0; i < task.smemU.size(); i++)
            warpAtomicAdd<true>(task.smemU.data() + i, U.data()[i]);
#endif
    }

    g.sync();
    assert(schur.upperU.data[task.idxItem].size() == task.smemU.size());
    for (uint32_t i = 0; i < divUp(task.smemU.size(), g.size()); i++) {
        const auto idx = g.size() * i + g.thread_rank();
        if (idx < task.smemU.size()) {
            schur.upperU.data[task.idxItem].data()[idx] = -task.smemU.data()[idx];//@info: note here we use negative smemU
        }
    }
}
}

template <typename Traits>
cudaError_t launchCudaComputeSchurUpperU(
        const Model<Traits, true> model,
        const VecVec<const SchurUpperUComputer::ObPair> taskList, // nbRows = nbUpperUItems
        const Hessian<Traits, true> hessian, //hessian is not needed. Put it here just for validation with assertions
        const typename HessianBase<Traits>::VSymBlock *__restrict__ invVList,
        const Schur<Traits, false> schur,
        const IdxUBlock *__restrict__ rowTableUpperU, // same structure as schur.upperU.idxCols
        const cudaStream_t stream) {
    if (taskList.getNbRows() == 0)
        return cudaGetLastError();

    constexpr unsigned ctaSize = 128;
    static_assert(ctaSize % warp_size == 0, "fatal error");
    const dim3 dimBlock(ctaSize);
    const dim3 dimGrid(divUp(taskList.getNbRows(), ctaSize / warp_size));
    return launchKernel(SchurUpperUComputer::compute<Traits, ctaSize>, dimGrid, dimBlock, 0, stream,
            model, taskList, hessian, invVList, schur, rowTableUpperU);
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchCudaComputeSchurUpperU(const Model<TRAITS, true> model,\
            const VecVec<const SchurUpperUComputer::ObPair> taskList,\
            const Hessian<TRAITS, true> hessian, const typename HessianBase<TRAITS>::VSymBlock *__restrict__ invVList,\
            const Schur<TRAITS, false> schur, const IdxUBlock *__restrict__ rowTableUpperU, const cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template<typename Traits, uint32_t ctaSize = 64>
__global__ void computeSchurDiagU(
        const Model<Traits, true> model,
        const VecVec<const uint16_t> sparseIdxW2idxLocalOb, // list of local ob index for variable points. nbRows = nbUpperUItems
        const Hessian<Traits, true> hessian,
        const typename HessianBase<Traits>::VSymBlock *__restrict__ invVList,
        const Schur<Traits, false> schur
) {
    RBA_IMPORT_TRAITS(Traits);
    constexpr unsigned grpSize = warp_size;
    constexpr unsigned grpsPerCta = ctaSize / grpSize;
    const auto g = cg::tiled_partition<grpSize>(cg::this_thread_block());
    assert(blockDim.x == ctaSize && blockDim.y == 1 && blockDim.z == 1);
    assert(gridDim.x * grpsPerCta >= schur.nbVarCaps && gridDim.y == 1 && gridDim.z == 1);
    const unsigned idxU = blockIdx.x * grpsPerCta + threadIdx.x / grpSize;
    if (idxU >= sparseIdxW2idxLocalOb.getNbRows())
        return;

    __shared__ struct {
        IdxUBlock idxU;
        const uint16_t *sparseIdxW2idxLocalOb;
        uint32_t nbVarObs;
        uint32_t idxCap;
        Capture capture;
        CamIntr camera;
        const CapOb<lpf> *obs;
        uint32_t nbObs;
        kmat<bool, 3> fixedCapLoc;
        mutable typename HessianBase<Traits>::USymBlock smemU;
        mutable typename HessianBase<Traits>::EaBlock smemEa;
    } tasks[grpsPerCta];
#if USE_TRANSPOSED_REDUCTION
#endif
    {
        auto &task = tasks[grpSize == ctaSize ? 0 : threadIdx.x / grpSize];
        if (g.thread_rank() == 0) {
            task.idxU = idxU;
            task.sparseIdxW2idxLocalOb = sparseIdxW2idxLocalOb[idxU];
            task.nbVarObs = sparseIdxW2idxLocalOb.getRowSize(idxU);
            assert(sparseIdxW2idxLocalOb.getNbRows() == schur.nbVarCaps);
            assert(task.nbVarObs == hessian.W.getRowLength(idxU));
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
        for (unsigned i = 0; i < divUp(task.smemU.size(), grpSize); i++) {
            const auto idx = grpSize * i + g.thread_rank();
            if (idx < task.smemU.size())
                task.smemU.data()[idx] = hessian.U[idxU].data()[idx];
        }
        assert(task.smemEa.size() <= grpSize);
        if (g.thread_rank() < task.smemEa.size()){
            task.smemEa.data()[g.thread_rank()] = hessian.Ea[idxU].data()[g.thread_rank()];
        }
    }
    g.sync();
    const auto &task = tasks[grpSize == ctaSize ? 0 : threadIdx.x / grpSize];
    uint16_t idxLocalOb[5];
    CapOb<lpf> obs[4];
    typename ModelBase<Traits>::PointWVarIdx points[3];
    typename HessianBase<Traits>::VSymBlock invVBlocks[2];
    typename HessianBase<Traits>::EbBlock blocksEb[2];

#pragma unroll
    for (uint32_t i = 1; i < 5; i++) {
        const uint32_t idxPre = g.size() * (i - 1) + g.thread_rank();
        const bool isInRange = idxPre < task.nbVarObs;
        if (i < 5) {
            idxLocalOb[i] = isInRange ? ldg(&task.sparseIdxW2idxLocalOb[idxPre]) : uint16_t(0xFFFFu);
        }
        if (i < 4) {
            if (isInRange) {
                assert(idxLocalOb[i] < task.nbObs);
                obs[i] = ldg(&task.obs[idxLocalOb[i]]);
            }
#ifndef NDEBUG
            else
                obs[i].ptIdx = ~0u;
#endif
        }
        if (i < 3) {
            if (isInRange) {
                points[i] = model.points[obs[i].ptIdx];
            }
#ifndef NDEBUG
            else
                points[i] = {{NAN, NAN, NAN}, ModelBase<Traits>::varIdxFixed};
#endif
        }
        if (i < 2) {
            if (isInRange) {
                assert(points[i].varIdx < model.varPoints.nbVar);
                invVBlocks[i] = invVList[points[i].varIdx];
                blocksEb[i] = hessian.Eb[points[i].varIdx];
            }
#ifndef NDEBUG
            else{
                for (unsigned e = 0; e < invVBlocks[i].size(); e++)
                    invVBlocks[i].data()[e] = NAN;
                blocksEb[i].assignScalar(NAN);
            }
#endif
        }
    }

    for (uint32_t n = 0; n < divUp(task.nbVarObs, g.size()); n++) {
        const uint32_t idx = g.size() * n + g.thread_rank();//sparseIdxW
#pragma unroll
        for (int i = 0; i < 5; i++) {
            const uint32_t idxPre = idx + g.size() * i;
            const bool isInRange = idxPre < task.nbVarObs;
            // rotate
            if (i < 4)
                idxLocalOb[i] = idxLocalOb[i + 1];
            if (i < 3)
                obs[i] = obs[i + 1];
            if (i < 2) {
                points[i] = points[i + 1];
            }
            if (i < 1) {
                invVBlocks[i] = invVBlocks[i + 1];
                blocksEb[i] = blocksEb[i + 1];
            }
            // prefetch
            if (i == 4)
                idxLocalOb[i] = isInRange ? task.sparseIdxW2idxLocalOb[idxPre] : uint16_t(0xFFFFu);
            if (i == 3) {
                if (isInRange) {
                    assert(idxLocalOb[i] < task.nbObs);
                    obs[i] = ldg(&task.obs[idxLocalOb[i]]);
                } else
                    obs[i].ptIdx = ~0u;
            }
            if (i == 2) {
                if (isInRange) {
                    points[i] = model.points[obs[i].ptIdx];
                }
#ifndef NDEBUG
                else
                    points[i] = {{NAN, NAN, NAN}, ModelBase<Traits>::varIdxFixed};
#endif
            }
            if (i == 1) {
                if (isInRange) {
                    assert(points[i].varIdx < model.varPoints.nbVar);
                    invVBlocks[i] = invVList[points[i].varIdx];
                    blocksEb[i] = hessian.Eb[points[i].varIdx];
                }
#ifndef NDEBUG
                else{
                    for (unsigned e = 0; e < invVBlocks[i].size(); e++)
                        invVBlocks[i].data()[e] = NAN;
                    blocksEb[i].assignScalar(NAN);
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
        if (idx < task.nbVarObs && hessian.W.data != nullptr) {
            const typename HessianBase<Traits>::WBlock W = errDeriv.jacobian.capture.transpose() * (weightedOmega * errDeriv.jacobian.pt);
            typename Traits::lpf absMaxW = 0.f;
            for (int i = 0; i < W.size(); i++) {
                absMaxW = std::max(absMaxW, std::abs(W.data()[i]));
            }
            for (int i = 0; i < W.size(); i++) {
                assert(std::abs(W.data()[i] - hessian.W[task.idxU][idx].data()[i]) <
                       0.01f * std::max(Traits::lpf(1.f)
                               , absMaxW));
            }
        }
#endif
        typename Schur<Traits, false>::USymBlock U = sqr(weightedOmega) * errDeriv.jacobian.capture.transpose() *
                                    (errDeriv.jacobian.pt * invVBlocks[0].toKMat() *
                                     errDeriv.jacobian.pt.transpose()) *
                                    errDeriv.jacobian.capture;
        typename Schur<Traits, false>::EaBlock Ea = weightedOmega * errDeriv.jacobian.capture.transpose() *
                                   (errDeriv.jacobian.pt * (invVBlocks[0].toKMat() * blocksEb[0]));
        if (idx >= task.nbVarObs) {
            U = Schur<Traits, false>::UBlock::zeros();
            Ea = Schur<Traits, false>::EaBlock::zeros();
        }
        assert(U.size() == task.smemU.size());
        for (int i = 0; i < task.smemU.size(); i++) {
            warpAtomicAdd<true>(task.smemU.data() + i, -U.data()[i]);//@info: note here we use negative
        }
        for(int i = 0; i < task.smemEa.size(); i++){
            warpAtomicAdd<true>(task.smemEa.data()+i, -Ea.data()[i]);//@info: note here we use negative
        }
    }

    g.sync();
    assert(schur.diagU[task.idxU].size() == task.smemU.size());
    for (uint32_t i = 0; i < divUp(task.smemU.size(), g.size()); i++) {
        const auto idx = g.size() * i + g.thread_rank();
        if (idx < task.smemU.size()) {
            schur.diagU[task.idxU].data()[idx] = task.smemU.data()[idx];//already applied negative when accumulating
//            if (idx == 0 && !task.smemU.allFinite()){
//                asm volatile("trap;\n");
//            }
        }
    }
    assert(schur.Ea[task.idxU].size() == task.smemEa.size());
    for (uint32_t i = 0; i < divUp(task.smemEa.size(), g.size()); i++) {
        const auto idx = g.size() * i + g.thread_rank();
        if (idx < task.smemEa.size()) {
            schur.Ea[task.idxU].data()[idx] = task.smemEa.data()[idx];//already applied negative when accumulating
//            if (idx == 0 && !task.smemEa.allFinite()){
//                asm volatile("trap;\n");
//            }
        }
    }
}

template <typename Traits>
cudaError launchCudaComputeSchurDiagU(
        const Model<Traits, true> model,
        const VecVec<const uint16_t> sparseIdxW2idxLocalOb, // list of local ob index for variable points.
        const Hessian<Traits, true> hessian,
        const typename HessianBase<Traits>::VSymBlock *__restrict__ invVList,
        const Schur<Traits, false> schur,
        const cudaStream_t stream) {
    constexpr unsigned ctaSize = 64;
    static_assert(ctaSize % warp_size == 0, "fatal error");
    const dim3 dimBlock(ctaSize);
    assert(schur.nbVarCaps == hessian.nbVarCaps && schur.nbVarCaps == model.involvedCaps.nbVar);
    const dim3 dimGrid(divUp(schur.nbVarCaps, ctaSize / warp_size));
    return launchKernel(computeSchurDiagU<Traits, ctaSize>, dimGrid, dimBlock, 0, stream,
            model, sparseIdxW2idxLocalOb, hessian, invVList, schur);
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError launchCudaComputeSchurDiagU(\
            const Model<TRAITS, true> model, const VecVec<const uint16_t> sparseIdxW2idxLocalOb,\
            const Hessian<TRAITS, true> hessian, const typename HessianBase<TRAITS>::VSymBlock *__restrict__ invVList,\
            const Schur<TRAITS, false> schur, const cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template <typename Traits, unsigned ctaSize = 64>
__global__ void computeSchurQ(
        const Model<Traits, true> model,
        const VecVec<const uint32_t> taskListSparseIdxS, // nbRows = nbQItems
        const VecVec<const uint16_t> taskListIdxLocalOb, //same shape as qPairSparseIdxS
        const IdxMBlock* rowTableQ, // idxRow with the same structure as schur.Q.idxCol
        const Hessian<Traits, true> hessian,
        const typename HessianBase<Traits>::VSymBlock* __restrict__ invVList,
        const Schur<Traits, false> schur)
{
    RBA_IMPORT_TRAITS(Traits);
    constexpr unsigned grpSize = warp_size;
    constexpr unsigned grpsPerCta = ctaSize / grpSize;
    const auto g = cg::tiled_partition<grpSize>((cg::this_thread_block()));
    assert(blockDim.x == ctaSize && blockDim.y == 1 && blockDim.z == 1);
    assert(gridDim.x * grpsPerCta >= taskListSparseIdxS.getNbRows() && gridDim.y == 1 && gridDim.z == 1);
    const uint32_t idxTask = blockIdx.x * grpsPerCta + threadIdx.x / grpSize;
    if (idxTask >= taskListSparseIdxS.getNbRows())
        return;

    __shared__ struct {
        uint32_t idxTask;
        IdxMBlock idxM;
        IdxUBlock idxU;
        const uint32_t* __restrict__ sparseIdxS;
        const uint16_t* __restrict__ idxLocalOb;
        uint32_t nbPairs;
        const typename HessianBase<Traits>::SBlock* __restrict__ sBlocks;
        uint32_t idxCap;
        Capture capture;
        CamIntr camera;
        IdxMBlock idxMIntri;
        const CapOb<lpf>* __restrict__ obs;
        uint32_t nbObs;
        kmat<bool, 3> fixedCapLoc;
        mutable typename HessianBase<Traits>::QBlock smemQ;
    } tasks[grpsPerCta];
#if USE_TRANSPOSED_REDUCTION
    constexpr unsigned sizeQ = HessianBase<Traits>::QBlock::size();
    using Reducer = GroupReducer<sizeQ, grpSize>;
    __shared__ typename Reducer::Buffer transBuffers[grpsPerCta];
#endif
    {
        auto& task = tasks[grpSize == ctaSize ? 0 : threadIdx.x / grpSize];
        if (g.thread_rank() == 0){
            task.idxTask = idxTask;
            assert(taskListSparseIdxS.getNbRows() == schur.Q.nbElems());
            task.idxM = rowTableQ[idxTask];
            task.idxU = schur.Q.idxCol[idxTask];
            assert(idxTask >= schur.Q.getRowBegin(task.idxM) && idxTask < schur.Q.getRowEnd(task.idxM));
            task.sparseIdxS = taskListSparseIdxS[idxTask];
            task.idxLocalOb = taskListIdxLocalOb[idxTask];
            task.nbPairs = taskListSparseIdxS.getRowSize(idxTask);
            assert(task.nbPairs == taskListIdxLocalOb.getRowSize(idxTask));
            task.sBlocks = hessian.S[task.idxM];
            const auto idxCap = model.involvedCaps.indices[task.idxU];
            task.idxCap = idxCap;
            task.capture = model.captures[idxCap];
            task.obs = model.capObs[idxCap];
            task.nbObs = model.capObs.getRowSize(idxCap);
            task.idxMIntri = isGroupModel<Traits>() ? model.cameras[task.capture.intrIdx].varIdx : ModelBase<Traits>::varIdxFixed;
            assert(task.idxMIntri == hessian.Q.row[task.idxU]);
        }
        g.sync();
        if (g.thread_rank() < 3) {
            task.fixedCapLoc[g.thread_rank()] = (model.gnssCtrl != nullptr) && model.gnssCtrl[task.idxCap].isHard(g.thread_rank());
        }
        static_assert(!isGroupModel<Traits>() || sizeof(task.camera) % sizeof(Traits::lpf) == 0 && sizeof(task.camera) / sizeof(Traits::lpf) < grpSize, "fatal error");
        if (isGroupModel<Traits>() && g.thread_rank() < sizeof(task.camera) / sizeof(Traits::lpf)) {
            reinterpret_cast<lpf *>(&task.camera)[g.thread_rank()] =
                    reinterpret_cast<const lpf *>(&model.cameras[task.capture.intrIdx].intri)[g.thread_rank()];
        }
        for (unsigned i = 0; i < divUp(task.smemQ.size(), g.size()); i++) {
            const unsigned idx = g.size() * i + g.thread_rank();
            if (idx < task.smemQ.size()) {
                if (task.idxMIntri == task.idxM) {
                    task.smemQ.data()[idx] = hessian.Q.blocks[task.idxU].data()[idx];
                } else {
                    task.smemQ.data()[idx] = 0;
                }
            }
        }
        g.sync();
    }
    const auto &task = tasks[grpSize == ctaSize ? 0 : threadIdx.x / grpSize];

    uint16_t idxLocalOb[5];
    CapOb<lpf> obs[4];
    typename ModelBase<Traits>::PointWVarIdx points[3];
    typename HessianBase<Traits>::VSymBlock invVBlocks[2];

    uint32_t sparseIdxS[3];
    typename HessianBase<Traits>::SBlock sBlocks[2];

#pragma unroll
    for (uint32_t i = 1; i < 5; i++) {
        const uint32_t idxPre = g.size() * (i - 1) + g.thread_rank();
        const bool isInRange = idxPre < task.nbPairs;
        if (i < 5) {
            if (isInRange){
                idxLocalOb[i] = task.idxLocalOb[idxPre];
            }
            else{
                idxLocalOb[i] = uint16_t(0xFFFFu);
            }
        }
        if (i < 4) {
            if (isInRange) {
                assert(idxLocalOb[i] < task.nbObs);
                obs[i] = task.obs[idxLocalOb[i]];
            }
            else
                obs[i].ptIdx = ~0u;
        }
        if (i < 3) {
            if (isInRange) {
                points[i] = model.points[obs[i].ptIdx];
                sparseIdxS[i] = task.sparseIdxS[idxPre];
            }
            else
                sparseIdxS[i] = ~0u;
        }
        if (i < 2) {
            if (isInRange) {
                assert(points[i].varIdx < model.varPoints.nbVar);
                invVBlocks[i] = invVList[points[i].varIdx];
                sBlocks[i] = task.sBlocks[sparseIdxS[i]];
            }
        }
    }

    for (uint32_t n = 0; n < divUp(task.nbPairs, g.size()); n++) {
        const uint32_t idx = g.size() * n + g.thread_rank();//sparseIdxW
#pragma unroll
        for (int i = 0; i < 5; i++) {
            const uint32_t idxPre = idx + g.size() * i;
            const bool isInRange = idxPre < task.nbPairs;
            // rotate
            if (i < 4)
                idxLocalOb[i] = idxLocalOb[i + 1];
            if (i < 3)
                obs[i] = obs[i + 1];
            if (i < 2) {
                points[i] = points[i + 1];
                sparseIdxS[i] = sparseIdxS[i + 1];
            }
            if (i < 1) {
                invVBlocks[i] = invVBlocks[i + 1];
                sBlocks[i] = sBlocks[i + 1];
            }
            // prefetch
            if (i == 4)
                idxLocalOb[i] = isInRange ? task.idxLocalOb[idxPre] : uint16_t(0xFFFFu);
            if (i == 3) {
                if (isInRange) {
                    assert(idxLocalOb[i] < task.nbObs);
                    obs[i] = task.obs[idxLocalOb[i]];
                }
                else
                    obs[i].ptIdx = ~0u;
            }
            if (i == 2) {
                if (isInRange) {
                    points[i] = model.points[obs[i].ptIdx];
                    sparseIdxS[i] = task.sparseIdxS[idxPre];
                }
#ifndef NDEBUG
                else{
                    points[i].varIdx = ModelBase<Traits>::varIdxFixed;
                    for(auto& c : points[i].pt.position)
                        c = locf{NAN};
                    sparseIdxS[i] = ~0u;
                }
#endif
            }
            if (i == 1) {
                if (isInRange) {
                    assert(points[i].varIdx < model.varPoints.nbVar);
                    invVBlocks[i] = invVList[points[i].varIdx];
                    sBlocks[i] = task.sBlocks[sparseIdxS[i]];
                }
#ifndef NDEBUG
                else{
                    sBlocks[i].assignScalar(NAN);
                }
#endif
            }
        }
        const auto errDeriv = computeErrorDerivative<Traits>(task.camera, task.capture,
            toKmat(points[0].pt.position), {obs[0].position},
            task.fixedCapLoc);
        const lpf weightedOmega = robustify<lpf>(errDeriv.error, obs[0].omega, obs[0].huber);
//#ifndef NDEBUG
//        // ref-check for W
//        if (idx < task.nbPairs && hessian.W.data != nullptr) {
//            const auto W = errDeriv.extri.capture.transpose() * (weightedOmega * errDeriv.jacobian.pt);
//            for (int i = 0; i < W.size(); i++)
//                assert(std::abs(W.data()[i] - hessian.W[task.idxU][...].data()[i]) < 1E-2f);
//        }
//#endif

        typename HessianBase<Traits>::QBlock Q = sBlocks[0] * invVBlocks[0].toKMat() * errDeriv.jacobian.pt.transpose() * weightedOmega * errDeriv.jacobian.capture;
        if (idx >= task.nbPairs) {
            Q = Schur<Traits, false>::QBlock::zeros();
        }
        assert(Q.size() == task.smemQ.size());
#if USE_TRANSPOSED_REDUCTION
        auto grpRedQ = kmat<typename HessianBase<Traits>::QBlock::ValType, divUp(sizeQ, grpSize)>::zeros();
        Reducer::reduce(g, transBuffers[grpSize == ctaSize ? 0 : threadIdx.x / grpSize], grpRedQ.data(), Q.data());
        for (int i = 0; i < divUp(sizeQ, grpSize); i++){
            if (grpSize*(i+1) <= sizeQ || g.thread_rank() < sizeQ % grpSize)
                atomicAdd(&task.smemQ.data()[grpSize*i+g.thread_rank()], -grpRedQ[i]);
        }
#else
        for (int i = 0; i < task.smemQ.size(); i++) {
            warpAtomicAdd<true>(task.smemQ.data() + i, -Q.data()[i]);//@info: note here we use negative
        }
#endif
    }
    g.sync();
    assert(schur.Q.data[task.idxTask].size() == task.smemQ.size());
    for (uint32_t i = 0; i < divUp(task.smemQ.size(), g.size()); i++) {
        const auto idx = g.size() * i + g.thread_rank();
        if (idx < task.smemQ.size())
            schur.Q.data[task.idxTask].data()[idx] = task.smemQ.data()[idx];//already applied negative when accumulating
    }
}

template <typename Traits>
cudaError_t launchCudaComputeSchurQ(
        const Model<Traits, true> model,
        const VecVec<const uint32_t> taskListSparseIdxS, // nbRows = nbQItems
        const VecVec<const uint16_t> taskListIdxLocalOb, //same shape as qPairSparseIdxS
        const IdxMBlock* rowTableQ, // idxRow with the same structure as schur.Q.idxCol
        const Hessian<Traits, true> hessian,
        const typename HessianBase<Traits>::VSymBlock* __restrict__ invVList,
        const Schur<Traits, false> schur,
        cudaStream_t stream)
{
    if (taskListSparseIdxS.getNbRows() == 0) {
        return cudaGetLastError();
    }
    constexpr unsigned ctaSize = 64;
    static_assert(ctaSize % warp_size == 0, "fatal error");
    const dim3 dimBlock(ctaSize);
    const dim3 dimGrid(divUp(taskListSparseIdxS.getNbRows(), ctaSize / warp_size));

    return launchKernel(computeSchurQ<Traits, ctaSize>, dimGrid, dimBlock, 0, stream,
            model, taskListSparseIdxS, taskListIdxLocalOb, rowTableQ, hessian, invVList, schur);
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchCudaComputeSchurQ(\
            const Model<TRAITS, true> model, const VecVec<const uint32_t> taskListSparseIdxS,\
            const VecVec<const uint16_t> taskListIdxLocalOb, const IdxMBlock* rowTableQ,\
            const Hessian<TRAITS, true> hessian, const typename HessianBase<TRAITS>::VSymBlock* __restrict__ invVList,\
            const Schur<TRAITS, false> schur, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template <typename Traits>
__global__ void kernel_computeInvV(typename HessianBase<Traits>::VSymBlock* __restrict__ dst, const typename HessianBase<Traits>::VSymBlock* __restrict__ src, uint32_t nbVBlocks)
{
    using VSymBlock = typename HessianBase<Traits>::VSymBlock;
    using VBlock = typename HessianBase<Traits>::VBlock;
    const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nbVBlocks)
        return;
    const VSymBlock V = src[idx];
#if 1
    auto circV = [&V](int i, int j){return V(i%V.rows(), j%V.cols());};
    VSymBlock minors;
    for (int i = 0; i < 3; i++){
        for (int j = 0; j <= i; j++){
            minors(j, i) = circV(i+1, j+1) * circV(i+2, j+2) - circV(i+2, j+1) * circV(i+1, j+2);
        }
    }
    const typename VSymBlock::ValType det = V(0, 0) * minors(0, 0) + V(0, 1) * minors(0, 1) + V(0, 2) * minors(0, 2);
    const VSymBlock invV = VSymBlock(1/det * minors.toKMat());
#else
	const VSymBlock invV = inverseByCholesky(V);
#endif
    dst[idx] = invV;

    // test in debug build
#ifndef NDEBUG
    {
        const VBlock error = V.toKMat() * invV.toKMat() - VBlock::eye();
        typename VBlock::ValType threshold = 1E-3f;
        for (unsigned i = 0; i < V.size(); i++) {
            threshold = std::max(threshold, 1E-3f * std::abs(V.data()[i]));
            threshold = std::max(threshold, 1E-3f * std::abs(invV.data()[i]));
        }
        for (uint32_t i = 0; i < error.size(); i++)
            assert(std::abs(error.data()[i]) < threshold);
    }
#endif
}

template <typename Traits>
cudaError_t launchCudaComputeInvV(typename HessianBase<Traits>::VSymBlock* __restrict__ dst, const typename HessianBase<Traits>::VSymBlock* __restrict__ src, uint32_t nbVBlocks, cudaStream_t stream)
{
    assert(dst != src);
    const unsigned ctaSize = 128;
    const unsigned gridSize = divUp(nbVBlocks, ctaSize);
    return launchKernel(kernel_computeInvV<Traits>, gridSize, ctaSize, 0, stream,
            dst, src, nbVBlocks);
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchCudaComputeInvV<TRAITS>(typename HessianBase<TRAITS>::VSymBlock* __restrict__ dst,\
            const typename HessianBase<TRAITS>::VSymBlock* __restrict__ src, uint32_t nbVBlocks, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template <typename Traits>
cudaError_t launchCudaComputeSchur(Model<Traits, true> model, Hessian<Traits, true> hessian, const typename HessianBase<Traits>::VSymBlock *invVList,
                                   Schur<Traits, false> schur, SchurPreComp precomp, cudaStream_t stream) {
    if (isGroupModel<Traits>()) {
        if (schur.nbVarIntri != 0) {
            checkEarlyReturn(cudaMemcpyAsync(schur.diagM, hessian.M,
                                             sizeof(typename HessianBase<Traits>::MSymBlock) * schur.nbVarIntri,
                                             cudaMemcpyDeviceToDevice, stream));
            checkEarlyReturn(cudaMemcpyAsync(schur.Ec, hessian.Ec,
                                             sizeof(typename HessianBase<Traits>::EcBlock) * schur.nbVarIntri,
                                             cudaMemcpyDeviceToDevice, stream));
        }
        const uint32_t nbSchurUpperMBlocks = precomp.upperM.pairSS.getNbRows();
        assert(nbSchurUpperMBlocks == schur.upperM.nbElems());
        if (schur.upperM.nbElems() != 0) {
            checkEarlyReturn(cudaMemsetAsync(schur.upperM.data, 0,
                                             sizeof(typename HessianBase<Traits>::MBlock) * nbSchurUpperMBlocks,
                                             stream));
        }
    }
#ifndef NDEBUG
    {
        std::vector<typename HessianBase<Traits>::USymBlock> diagUData(schur.nbVarCaps, typename HessianBase<Traits>::USymBlock(
                HessianBase<Traits>::UBlock::zeros() * typename Traits::hpf(NAN)));
        std::vector<typename HessianBase<Traits>::EaBlock> EaData(schur.nbVarCaps,
                                                 HessianBase<Traits>::EaBlock::zeros() * typename Traits::hpf(NAN));
        checkCudaErrors(cudaMemcpyAsync(schur.diagU, diagUData.data(), sizeof(typename HessianBase<Traits>::USymBlock) * schur.nbVarCaps,
                                        cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(schur.Ea, EaData.data(), sizeof(typename HessianBase<Traits>::EaBlock) * schur.nbVarCaps,
                                        cudaMemcpyHostToDevice, stream));

        const uint32_t nbSchurUpperUBlocks = precomp.upperU.pairs.getNbRows();
        std::vector<typename HessianBase<Traits>::UBlock> upperUData(nbSchurUpperUBlocks, HessianBase<Traits>::UBlock::zeros() *
                                                                         typename Traits::hpf(
                                                                                 NAN));//schur.upperU.nbElems() is in managed memory.
        checkCudaErrors(
                cudaMemcpyAsync(schur.upperU.data, upperUData.data(), sizeof(typename HessianBase<Traits>::UBlock) * nbSchurUpperUBlocks,
                                cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
#endif

    //invVList should be pre-computed
//    checkEarlyReturn(launchCudaComputeInvV(const_cast<typename HessianBase<Traits>::VSymBlock *>(invVList), hessian.V, hessian.nbVarPts, stream));
    checkEarlyReturn(launchCudaComputeSchurUpperM(hessian, invVList, schur, precomp.upperM, stream));
    checkEarlyReturn(launchCudaComputeSchurDiagM(hessian, invVList, schur, stream));
    checkEarlyReturn(launchCudaComputeSchurUpperU(model, precomp.upperU.pairs, hessian, invVList, schur, precomp.upperU.rowTableUpperU, stream));
    checkEarlyReturn(launchCudaComputeSchurDiagU(model, precomp.sparseIdxW2idxLocalOb, hessian, invVList, schur, stream));
    checkEarlyReturn(launchCudaComputeSchurQ(model, precomp.Q.taskListSparseIdxS, precomp.Q.taskListIdxLocalOb, precomp.Q.rowTableQ, hessian, invVList, schur, stream));
    return cudaGetLastError();
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchCudaComputeSchur(Model<TRAITS, true> model, Hessian<TRAITS, true> hessian, const typename HessianBase<TRAITS>::VSymBlock *invVList,\
                                                Schur<TRAITS, false> schur, SchurPreComp precomp, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

}//namespace rba
//@fixme: do not forget to memset/init schur.