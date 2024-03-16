//
// Created by yao on 1/12/18.
//

#include "../cuda_hint.cuh"
#include "solveSchurPCG.h"
#include <cooperative_groups.h>
#include <boost/preprocessor/seq/for_each.hpp>
namespace cg = cooperative_groups;

namespace rba{
namespace pcg{
//compute schur*a
namespace SchurMMV {
// this kernel also work as initialization for dstA, but dstB should be pre-initialized before this kernel
template<typename Traits, typename SymBlockType, typename VecType, bool computeDotProduct>
__global__ void computeDiag(
        VecType* __restrict__ dstA, typename Traits::epf* __restrict__ dstB,
        const SymBlockType* __restrict__ diagBlocks,
        const VecType* __restrict__ vecA, const VecType* __restrict__ vecB,
        uint32_t nbBlocks)
{
    static_assert(std::is_same<typename SymBlockType::ValType, typename VecType::ValType>::value, "fatal error");
    const unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nbBlocks)
        return;
    const SymBlockType m = diagBlocks[idx];
    const VecType a = vecA[idx];
    const VecType vec = m.toKMat() * a;
    dstA[idx] = vec;
    if (computeDotProduct){
        atomicAdd(dstB, (vecB[idx].transpose()*vec)[0]);
    }
}
// compute diag*vecA and optionally vecB*diag*vecA
template <typename Traits>
cudaError_t launchForDiag(
        const SchurVec<Traits, false>& dstVec, typename Traits::epf* dstScalar,
        const SchurDiag<Traits, true>& diag, const SchurVec<Traits, true>& vecA, const SchurVec<Traits, true>& vecB, cudaStream_t stream)
{
    cudaError_t err;
    if (dstScalar){
        err = cudaMemsetAsync(dstScalar, 0, sizeof(*dstScalar), stream);
        if (err != cudaSuccess)
            return err;
    }
    if (isGroupModel<Traits>() && dstVec.nbCBlocks != 0)
    {
        assert(dstVec.nbCBlocks == diag.nbMBlocks && dstVec.nbCBlocks == vecA.nbCBlocks);
        if(dstScalar != nullptr)
            assert(dstVec.nbCBlocks == vecB.nbCBlocks);
        uint32_t dimBlock = 128;
        uint32_t dimGrid = divUp(dstVec.nbCBlocks, dimBlock);
        if (dstScalar != nullptr)
            computeDiag<Traits, typename HessianBase<Traits>::MSymBlock, typename HessianBase<Traits>::EcBlock, true><<<dimGrid, dimBlock, 0, stream>>>(
                    dstVec.c, dstScalar, diag.M, vecA.c, vecB.c, dstVec.nbCBlocks);
        else
            computeDiag<Traits, typename HessianBase<Traits>::MSymBlock, typename HessianBase<Traits>::EcBlock, false><<<dimGrid, dimBlock, 0, stream>>>(
                    dstVec.c, dstScalar, diag.M, vecA.c, vecB.c, dstVec.nbCBlocks);
        checkEarlyReturn(cudaGetLastError());
    }
    if (dstVec.nbABlocks != 0)
    {
        assert(dstVec.nbABlocks == diag.nbUBlocks && dstVec.nbABlocks == vecA.nbABlocks);
        if(dstScalar != nullptr)
            assert(dstVec.nbABlocks == vecB.nbABlocks);
        uint32_t dimBlock = 128;
        uint32_t dimGrid = divUp(dstVec.nbABlocks, dimBlock);
        if (dstScalar != nullptr)
            computeDiag<Traits, typename HessianBase<Traits>::USymBlock, typename HessianBase<Traits>::EaBlock, true><<<dimGrid, dimBlock, 0, stream>>>(
                    dstVec.a, dstScalar, diag.U, vecA.a, vecB.a, dstVec.nbABlocks);
        else
            computeDiag<Traits, typename HessianBase<Traits>::USymBlock, typename HessianBase<Traits>::EaBlock, false><<<dimGrid, dimBlock, 0, stream>>>(
                    dstVec.a, dstScalar, diag.U, vecA.a, vecB.a, dstVec.nbABlocks);
        checkEarlyReturn(cudaGetLastError());
    }
    return cudaGetLastError();
}

#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchForDiag(\
        const SchurVec<TRAITS, false>& dstVec, typename TRAITS::epf* dstScalar,\
        const SchurDiag<TRAITS, true>& diag, const SchurVec<TRAITS, true>& vecA,\
        const SchurVec<TRAITS, true>& vecB, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

//using BlockType = typename HessianBase<Traits>::MBlock;
//using AccVecType = typename HessianBase<Traits>::EcBlock;
//constexpr unsigned grpSize = 32;
//constexpr unsigned ctaGrps = 4;
template<typename BlockType, typename VecType, unsigned grpSize = 32, unsigned ctaGrps = 4>
__global__ void computeUpperMU(
        VecType* __restrict__ dst,
        CSR<const BlockType> blocks, const VecType* __restrict__ vectors)
{
    assert(blockDim.x == grpSize && blockDim.y == ctaGrps && blockDim.z == 1);
    assert(gridDim.x == 1 && gridDim.y == divUp(blocks.nbRows, ctaGrps) && gridDim.z == 1);
    const auto g = cg::tiled_partition<grpSize>(cg::this_thread_block());
    const unsigned idxRow = ctaGrps * blockIdx.y + threadIdx.y;
    if (idxRow >= blocks.nbRows)
        return;
    __shared__ struct {
        uint32_t idxRow;
        uint32_t rowBeg;
        uint32_t rowSize;
        VecType smemAcc;
    } tasks[ctaGrps];
    auto& task = tasks[threadIdx.y];
    if (g.thread_rank() == 0){
        task.idxRow = idxRow;
        task.rowBeg = blocks.getRowBegin(idxRow);
        task.rowSize = blocks.getRowEnd(idxRow) - task.rowBeg;
    }
    assert(task.smemAcc.size() < grpSize);
    if (g.thread_rank() < task.smemAcc.size())
        task.smemAcc.data()[g.thread_rank()] = 0;
    g.sync();

    for (uint32_t n = 0; n < divUp(task.rowSize, grpSize); n++){
        const uint32_t idx = grpSize * n + g.thread_rank();
        VecType result;
        if (idx < task.rowSize) {
            const uint32_t idxCol = blocks.idxCol[task.rowBeg + idx];
            const BlockType block = blocks.data[task.rowBeg + idx];
            const VecType vec = vectors[idxCol];
            result = block * vec;
        }
        else{
            result = VecType::zeros();
        }
        for (uint32_t i = 0; i < result.size(); i++) {
            warpAtomicAdd<true, VecType::ValType, VecType::ValType>(&task.smemAcc.data()[i], result.data()[i]);
        }
    }
    g.sync();
    assert(task.smemAcc.size() < grpSize);
    if (g.thread_rank() < task.smemAcc.size()) {
          atomicAdd(&dst[task.idxRow].data()[g.thread_rank()], task.smemAcc.data()[g.thread_rank()]);
    }
}

template <typename Traits>
cudaError_t launchForUpperM(typename HessianBase<Traits>::EcBlock* dst,
        const CSR<const typename HessianBase<Traits>::MBlock>& upperM,
        const typename HessianBase<Traits>::EcBlock* Ec, cudaStream_t stream)
{
    if (upperM.nbRows == 0){
        return cudaGetLastError();
    }
    constexpr dim3 dimBlock = {32, 4};
    const dim3 dimGrid = {1, divUp(upperM.nbRows, dimBlock.y)};
    return launchKernel(
            computeUpperMU<typename HessianBase<Traits>::MBlock, typename HessianBase<Traits>::EcBlock, dimBlock.x, dimBlock.y>,
                    dimGrid, dimBlock, 0, stream, dst, upperM, Ec);
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchForUpperM<TRAITS>(typename HessianBase<TRAITS>::EcBlock* dst,\
        const CSR<const typename HessianBase<TRAITS>::MBlock>& upperM,\
        const typename HessianBase<TRAITS>::EcBlock* Ec, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template <typename Traits>
cudaError_t launchForUpperU(typename HessianBase<Traits>::EaBlock* dst, const CSR<const typename HessianBase<Traits>::UBlock>& upperU,
                            const typename HessianBase<Traits>::EaBlock* Ea, cudaStream_t stream)
{
    if (upperU.nbRows == 0) {
        return cudaGetLastError();
    }
    constexpr dim3 dimBlock = {32, 4};
    const dim3 dimGrid = {1, divUp(upperU.nbRows, dimBlock.y)};
    return launchKernel(
            computeUpperMU<typename HessianBase<Traits>::UBlock, typename HessianBase<Traits>::EaBlock, dimBlock.x, dimBlock.y>,
            dimGrid, dimBlock, 0, stream, dst, upperU, Ea);
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchForUpperU<TRAITS>(typename HessianBase<TRAITS>::EaBlock* dst, const CSR<const typename HessianBase<TRAITS>::UBlock>& upperU,\
        const typename HessianBase<TRAITS>::EaBlock* Ea, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template<typename BlockType, typename VecType, unsigned grpSize = 32, unsigned ctaGrps = 4>
__global__ void computeLowerMU(
        VecType* __restrict__ dst,
        CSR<const uint32_t> blocks, CSR<const BlockType> upper,
        const VecType* __restrict__ vectors)
{
    assert(blockDim.x == grpSize && blockDim.y == ctaGrps && blockDim.z == 1);
    assert(gridDim.x == 1 && gridDim.y == divUp(blocks.nbRows, ctaGrps) && gridDim.z == 1);
    const auto g = cg::tiled_partition<grpSize>(cg::this_thread_block());
    const unsigned idxRow = ctaGrps * blockIdx.y + threadIdx.y;
    if (idxRow >= blocks.nbRows)
        return;
    __shared__ struct {
        uint32_t idxRow;
        uint32_t rowBeg;
        uint32_t rowSize;
        VecType smemAcc;
    } tasks[ctaGrps];
    auto& task = tasks[threadIdx.y];
    if (g.thread_rank() == 0){
        task.idxRow = idxRow;
        task.rowBeg = blocks.getRowBegin(idxRow);
        task.rowSize = blocks.getRowEnd(idxRow) - task.rowBeg;
    }
    assert(task.smemAcc.size() < grpSize);
    if (g.thread_rank() < task.smemAcc.size())
        task.smemAcc.data()[g.thread_rank()] = 0;
    g.sync();

    for (uint32_t n = 0; n < divUp(task.rowSize, grpSize); n++){
        const uint32_t idx = grpSize * n + g.thread_rank();
        VecType result;
        if (idx < task.rowSize) {
            const uint32_t idxCol = ldg(&blocks.idxCol[task.rowBeg + idx]);
            assert(idxRow == upper.idxCol[blocks.data[task.rowBeg + idx]]);
            const BlockType block = upper.data[ldg(&blocks.data[task.rowBeg + idx])].transpose();
            const VecType vec = vectors[idxCol];
            result = block * vec;
        }
        else{
            result = VecType::zeros();
        }
#pragma unroll
        for (uint32_t i = 0; i < decltype(result)::size(); i++) {
            warpAtomicAdd<true, VecType::ValType, VecType::ValType>(&task.smemAcc.data()[i], result.data()[i]);
        }
    }
    g.sync();
    assert(task.smemAcc.size() < grpSize);
    if (g.thread_rank() < task.smemAcc.size()) {
        atomicAdd(&dst[task.idxRow].data()[g.thread_rank()], task.smemAcc.data()[g.thread_rank()]);
    }
}
template <typename Traits>
cudaError_t launchForLowerM(typename HessianBase<Traits>::EcBlock* dst, const CSR<const uint32_t>& lowerM,
                            const CSR<const typename HessianBase<Traits>::MBlock>& upperM,
                            const typename HessianBase<Traits>::EcBlock* Ec, cudaStream_t stream)
{
    if (lowerM.nbRows == 0) {
        return cudaGetLastError();
    }
    constexpr dim3 dimBlock = {32, 4};
    const dim3 dimGrid = {1, divUp(lowerM.nbRows, dimBlock.y)};
    computeLowerMU<<<dimGrid, dimBlock, 0, stream>>>(dst, lowerM, upperM, Ec);
    return cudaGetLastError();
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchForLowerM<TRAITS>(typename HessianBase<TRAITS>::EcBlock* dst, const CSR<const uint32_t>& lowerM,\
                                        const CSR<const typename HessianBase<TRAITS>::MBlock>& upperM,\
                                        const typename HessianBase<TRAITS>::EcBlock* Ec, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template <typename Traits>
cudaError_t launchForLowerU(typename HessianBase<Traits>::EaBlock* dst, const CSR<const uint32_t>& lowerU,
                            const CSR<const typename HessianBase<Traits>::UBlock>& upperU,
                            const typename HessianBase<Traits>::EaBlock* Ea, cudaStream_t stream)
{
    if (lowerU.nbRows == 0) {
        return cudaGetLastError();
    }
    const dim3 dimBlock = {32, 4};
    const dim3 dimGrid = {1, divUp(lowerU.nbRows, dimBlock.y)};
    computeLowerMU<<<dimGrid, dimBlock, 0, stream>>>(dst, lowerU, upperU, Ea);
    return cudaGetLastError();
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchForLowerU<TRAITS>(typename HessianBase<TRAITS>::EaBlock* dst, const CSR<const uint32_t>& lowerU,\
                                const CSR<const typename HessianBase<TRAITS>::UBlock>& upperU,\
                                const typename HessianBase<TRAITS>::EaBlock* Ea, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template<typename Traits, typename DstVecType = typename HessianBase<Traits>::EcBlock,
        typename BlockType = typename HessianBase<Traits>::QBlock,
        typename VecType = typename HessianBase<Traits>::EaBlock>
__global__ void computeUpperQ(
        DstVecType* __restrict__ dst, CSR<const BlockType> blocks, const VecType* __restrict__ vectors)
{
    assert(blockDim.y == 1 && blockDim.z == 1);
    const uint2 idx = {
            blockDim.x * blockIdx.x + threadIdx.x,
            blockIdx.y
    };
    assert(gridDim.y == blocks.nbRows&& gridDim.z == 1);
    __shared__ uint32_t iters;
    __shared__ uint32_t ctaItems;
    __shared__ uint32_t ctaRowBeg;
    __shared__ DstVecType smemAcc;
    if(threadIdx.x == 0){
        const uint32_t rowBeg = blocks.getRowBegin(idx.y);
        const uint32_t rowLength = blocks.getRowLength(idx.y);
        iters = divUp(rowLength, blockDim.x * gridDim.x);
        const uint32_t targetCtaItems = blockDim.x * iters;
        ctaRowBeg = targetCtaItems * blockIdx.x;
        ctaItems = std::min(targetCtaItems, ctaRowBeg < rowLength ? rowLength - ctaRowBeg : 0u);
        ctaRowBeg += rowBeg;
    }
    assert(smemAcc.size() < blockDim.x);
    if (threadIdx.x < smemAcc.size())
        smemAcc.data()[threadIdx.x] = 0;
    __syncthreads();
    if (ctaItems == 0)
        return;
    for (unsigned n = 0; n < iters; n++){
        const uint32_t idxCtaItem = blockDim.x * n + threadIdx.x;
		DstVecType dstVec;
        if (idxCtaItem < ctaItems) {
			const uint32_t idxCol = blocks.idxCol[ctaRowBeg + idxCtaItem];
			const BlockType block = blocks.data[ctaRowBeg + idxCtaItem];
			const VecType vec = vectors[idxCol];
			dstVec = (block * vec).template cast<typename DstVecType::ValType>();
		}
		else {
			dstVec = DstVecType::zeros();
		}
        //@todo: take thread sum first
        for (uint32_t i = 0; i < dstVec.size(); i++)
            warpAtomicAdd<true>(&smemAcc.data()[i], dstVec.data()[i]);
    }
    __syncthreads();
    assert(smemAcc.size() < blockDim.x);
    if (threadIdx.x < smemAcc.size())
        atomicAdd(&dst[idx.y].data()[threadIdx.x], smemAcc.data()[threadIdx.x]);
}
template <typename Traits>
cudaError_t launchForUpperQ(
        typename HessianBase<Traits>::EcBlock * __restrict__ dst, CSR<const typename HessianBase<Traits>::QBlock> blocks,
        const typename HessianBase<Traits>::EaBlock * __restrict__ vectors,
        cudaStream_t stream)
{
    if (blocks.nbRows == 0){
        return cudaGetLastError();
    }
    dim3 dimBlock(128);
    dim3 dimGrid(4, blocks.nbRows);
    computeUpperQ<Traits><<<dimGrid, dimBlock, 0, stream>>>(dst, blocks, vectors);
    return cudaGetLastError();
}

#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchForUpperQ<TRAITS>(\
            typename HessianBase<TRAITS>::EcBlock * __restrict__ dst, CSR<const typename HessianBase<TRAITS>::QBlock> blocks,\
            const typename HessianBase<TRAITS>::EaBlock * __restrict__ vectors,\
            cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template<typename Traits, typename DstVecType = typename HessianBase<Traits>::EaBlock, typename BlockType = typename HessianBase<Traits>::QBlock,
        typename VecType = typename HessianBase<Traits>::EcBlock>
__global__ void computeLowerQ(
        DstVecType* __restrict__ dst, CSR<const uint32_t> blocks, CSR<const BlockType> upper,
        const VecType* __restrict__ vectors)
{
    assert(blockDim.y == 1 && blockDim.z == 1);
    const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= blocks.nbRows)
        return;
    const uint32_t rowBeg = blocks.getRowBegin(idx);
    const uint32_t rowSize = blocks.getRowLength(idx);
    DstVecType dstVec = DstVecType::zeros();
    for (uint32_t n = 0; n < rowSize; n++){
        const uint32_t idxCol = blocks.idxCol[rowBeg + n];
        const auto block = upper.data[blocks.data[rowBeg + n]].transpose();
        const VecType vec = vectors[idxCol];
        dstVec = dstVec + block.template cast<typename DstVecType::ValType>() * vec.template cast<typename DstVecType::ValType>();
    }
    for (uint32_t i = 0; i < dstVec.size(); i++)
        atomicAdd(&dst[idx].data()[i], dstVec.data()[i]);
}
template <typename Traits>
cudaError_t launchForLowerQ(
        typename HessianBase<Traits>::EaBlock * __restrict__ dst, CSR<const uint32_t> blocks, CSR<const typename HessianBase<Traits>::QBlock> upper,
        const typename HessianBase<Traits>::EcBlock * __restrict__ vectors,
        cudaStream_t stream)
{
    if (blocks.nbRows == 0){
        return cudaGetLastError();
    }
    dim3 dimBlock(128);
    dim3 dimGrid(divUp(blocks.nbRows, dimBlock.x));
    computeLowerQ<Traits><<<dimGrid, dimBlock, 0, stream>>>(dst, blocks, upper, vectors);
    return cudaGetLastError();
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchForLowerQ<TRAITS>(\
            typename HessianBase<TRAITS>::EaBlock * __restrict__ dst, CSR<const uint32_t> blocks,\
            CSR<const typename HessianBase<TRAITS>::QBlock> upper,\
            const typename HessianBase<TRAITS>::EcBlock * __restrict__ vectors,\
            cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

}//namespace SchurMMV

namespace SchurDotProd{
template <typename VecType, typename AccType>
__global__ void compute(AccType* __restrict__ acc,
        const VecType* __restrict__ vecA, const VecType* __restrict__ vecB,
        uint32_t nbVecs)
{
    const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    const auto val = idx < nbVecs ? (vecA[idx].transpose() * vecB[idx])[0] : 0.f;
    warpAtomicAdd<true>(acc, val);
}
template <typename Traits>
cudaError_t launch(typename Traits::epf* __restrict__ acc,
                   const SchurVec<Traits, true>& vecA, const SchurVec<Traits, true>& vecB,
                   cudaStream_t stream)
{
    using epf = typename Traits::epf;
    checkEarlyReturn(cudaMemsetAsync(acc, 0, sizeof(*acc), stream));
    if (vecA.nbCBlocks != 0)
    {
        assert(vecA.nbCBlocks == vecB.nbCBlocks);
        unsigned dimBlock = 128;
        unsigned dimGrid = divUp(vecA.nbCBlocks, dimBlock);
        compute<typename HessianBase<Traits>::EcBlock, epf> << <dimGrid, dimBlock, 0, stream>>>(acc, vecA.c, vecB.c, vecA.nbCBlocks);
        checkEarlyReturn(cudaGetLastError());
    }
    if (vecA.nbABlocks != 0)
    {
        assert(vecA.nbABlocks == vecB.nbABlocks);
        unsigned dimBlock = 128;
        unsigned dimGrid = divUp(vecA.nbABlocks, dimBlock);
        compute<typename HessianBase<Traits>::EaBlock, epf><<<dimGrid, dimBlock, 0, stream>>>(acc, vecA.a, vecB.a, vecA.nbABlocks);
        checkEarlyReturn(cudaGetLastError());
    }
    return cudaGetLastError();
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launch(typename TRAITS::epf* __restrict__ acc,\
                    const SchurVec<TRAITS, true>& vecA, const SchurVec<TRAITS, true>& vecB,\
                    cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES
}//namespace SchurDotProd

namespace SchurVecAdd{
//compute alpha*vecA+vecB. Not using __restrict__ to allow aliasing
template <typename VecType, typename AlphaType>
__global__ void compute(VecType* dst,// do not using __restrict__ here!
        const AlphaType* alpha, bool negativeAlpha, const VecType* vecA,
        const VecType* vecB, uint32_t nbVecs)
{
    const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nbVecs)
        return;
    __shared__ typename VecType::ValType smemAlpha;
    if (threadIdx.x == 0)
        smemAlpha = typename VecType::ValType(*alpha * (negativeAlpha ? -1 : 1));
    __syncthreads();
    dst[idx] = smemAlpha * vecA[idx] + vecB[idx];
}
template <typename Traits>
cudaError_t launch(const SchurVec<Traits, false>& dst, const typename Traits::epf* alpha, const bool negativeAlpha,
        const SchurVec<Traits, true>& vecA, const SchurVec<Traits, true>& vecB, cudaStream_t stream)
{
    if (isGroupModel<Traits>() && dst.nbCBlocks != 0)
    {
        assert(dst.nbCBlocks == vecA.nbCBlocks && dst.nbCBlocks == vecB.nbCBlocks);
        unsigned dimBlock = 128;
        unsigned dimGrid = divUp(dst.nbCBlocks, dimBlock);
        compute<typename HessianBase<Traits>::EcBlock> << <dimGrid, dimBlock, 0, stream>>>(dst.c, alpha, negativeAlpha, vecA.c, vecB.c, dst.nbCBlocks);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }
    if (dst.nbABlocks != 0)
    {
        assert(dst.nbABlocks == vecA.nbABlocks && dst.nbABlocks == vecB.nbABlocks);
        unsigned dimBlock = 128;
        unsigned dimGrid = divUp(dst.nbABlocks, dimBlock);
        compute<typename HessianBase<Traits>::EaBlock><<<dimGrid, dimBlock, 0, stream>>>(dst.a, alpha, negativeAlpha, vecA.a, vecB.a, dst.nbABlocks);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }
    return cudaGetLastError();
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launch(const SchurVec<TRAITS, false>& dst, const typename TRAITS::epf* alpha, const bool negativeAlpha,\
                                const SchurVec<TRAITS, true>& vecA, const SchurVec<TRAITS, true>& vecB, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES
}//namespace SchurVecAdd

template <typename SymBlock, bool useDouble = false, bool useNaNHack = true>
__global__ void computeInverse(SymBlock* __restrict__ dst, const SymBlock* __restrict__ src, uint32_t nbBlocks)
{
    const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nbBlocks) {
    	using ArithmeticType = typename std::conditional<useDouble, double, typename SymBlock::ValType>::type;
    	auto s = src[idx].template cast<ArithmeticType>();
        SymBlock tmp = inverseByCholesky(s).template cast<typename SymBlock::ValType>();
        //@fixme: This is a temporary hack for NaN. It should be fixed either: 1. by better model filter. 2. by using a per-block lambda
        if (useNaNHack && !tmp.allFinite()){
            typename SymBlock::ValType diagMax = 0.f;
#pragma unroll
            for (uint32_t i = 0; i < SymBlock::rows(); i++){
                diagMax = std::max(diagMax, std::abs(s(i, i)));
            }
            typename SymBlock::ValType diagVal = diagMax;
#pragma unroll(1)
            for (int n = 0; n < 24; n++){
#pragma unroll
                for (uint32_t i = 0; i < SymBlock::rows(); i++){
                    s(i, i) = diagVal;
                }
                tmp = inverseByCholesky(s).template cast<typename SymBlock::ValType>();
                diagVal *= 2;
                if (!tmp.allFinite()){
                    break;
                }
            }
        }
        dst[idx] = tmp;
    }
}

template <typename ftype, uint32_t nbRows>
cudaError_t launchCudaComputeInverse(symkmat<ftype, nbRows>* __restrict__ dst,
        const symkmat<ftype, nbRows>* __restrict__ src, uint32_t nbBlocks, cudaStream_t stream)
{
    if (nbBlocks == 0u){
        return cudaGetLastError();
    }
    const uint32_t dimBlock = 64;
    const uint32_t dimGrid = divUp(nbBlocks, dimBlock);
    computeInverse<<<dimGrid, dimBlock, 0, stream>>>(dst, src, nbBlocks);
    return cudaGetLastError();
}
template cudaError_t launchCudaComputeInverse(
    symkmat<double, 9>*, const symkmat<double, 9>*, uint32_t, cudaStream_t);
template cudaError_t launchCudaComputeInverse(
    symkmat<double, 0>*, const symkmat<double, 0>*, uint32_t, cudaStream_t);
template cudaError_t launchCudaComputeInverse(
    symkmat<float, 9>*, const symkmat<float, 9>*, uint32_t, cudaStream_t);
template cudaError_t launchCudaComputeInverse(
    symkmat<double, 8>*, const symkmat<double, 8>*, uint32_t, cudaStream_t);
template cudaError_t launchCudaComputeInverse(
    symkmat<double, 7>*, const symkmat<double, 7>*, uint32_t, cudaStream_t); // F2D5
template cudaError_t launchCudaComputeInverse(
    symkmat<double, 6>*, const symkmat<double, 6>*, uint32_t, cudaStream_t); // F1D5
template cudaError_t launchCudaComputeInverse(
    symkmat<double, 3>*, const symkmat<double, 3>*, uint32_t, cudaStream_t);
template cudaError_t launchCudaComputeInverse(
    symkmat<double, 1>*, const symkmat<double, 1>*, uint32_t, cudaStream_t);
template cudaError_t launchCudaComputeInverse(
    symkmat<double, 2>*, const symkmat<double, 2>*, uint32_t, cudaStream_t);
template cudaError_t launchCudaComputeInverse(
    symkmat<double, 4>*, const symkmat<double, 4>*, uint32_t, cudaStream_t);
template cudaError_t launchCudaComputeInverse(
    symkmat<float, 6>*, const symkmat<float, 6>*, uint32_t, cudaStream_t);
template cudaError_t launchCudaComputeInverse(
    symkmat<float, 7>*, const symkmat<float, 7>*, uint32_t, cudaStream_t);

template <typename ftype>
__global__ void kernel_updateBeta(ftype& __restrict__ beta,
        ftype(& __restrict__ zTr)[2]){
    beta = zTr[1] / zTr[0];
    zTr[0] = zTr[1];
    if (!std::isfinite(beta)){
        beta = 0.f;
    }
}
template <typename ftype>
cudaError_t launchUpdateBeta(ftype& beta, ftype(&zTr)[2], cudaStream_t stream)
{
    kernel_updateBeta<<<1,1,0,stream>>>(beta, zTr);
    return cudaGetLastError();
}
template cudaError_t launchUpdateBeta(double& beta, double(&zTr)[2], cudaStream_t stream);

template <typename ftype>
__global__ void kernel_updateAlpha(ftype& alpha,
        const ftype& zTr, const ftype& pTAp){
    alpha = zTr / pTAp;
    if (!std::isfinite(alpha)){
        assert(pTAp == 0);
        alpha = 0.f;
    }
}
template <typename ftype>
cudaError_t launchUpdateAlpha(ftype& alpha, const ftype& zTr, const ftype& pTAp, cudaStream_t stream)
{
    kernel_updateAlpha<<<1,1,0,stream>>>(alpha, zTr, pTAp);
    return cudaGetLastError();
}
template cudaError_t launchUpdateAlpha(double& alpha, const double& zTr, const double& pTAp, cudaStream_t stream);

namespace CheckThreshold {
template <typename T, uint32_t Rows, uint32_t Cols>
__device__ __forceinline__
void checkItem(uint32_t *__restrict__ counter, const kmat<T, Rows, Cols>* __restrict__ data,
        uint32_t idx, const kmat<T, Rows, Cols>& threshold){
    const kmat<T, Rows, Cols> r = data[idx];
    for (unsigned i = 0; i < r.size(); i++) {
        assert(threshold.data()[i] >= 0);
        if (std::abs(r.data()[i]) > threshold.data()[i]) {
            atomicAdd(counter, 1);
            break;
        }
    }
}
// count above threshold
template <typename Traits>
__global__ void kernel(uint32_t *__restrict__ counter, SchurVec<Traits, true> residue, typename HessianBase<Traits>::EcBlock c,
        typename HessianBase<Traits>::EaBlock a) {
    constexpr uint32_t ctaSize = 128;
    assert(blockDim.x == ctaSize);
    const uint32_t nbCtaC = divUp(residue.nbCBlocks, ctaSize);
    if (blockIdx.x < nbCtaC) {
        const uint32_t idx = ctaSize * blockIdx.x + threadIdx.x;
        if (idx >= residue.nbCBlocks)
            return;
        checkItem(counter, residue.c, idx, c);
    } else {
        const uint32_t idx = ctaSize * (blockIdx.x - nbCtaC) + threadIdx.x;
        if (idx >= residue.nbABlocks)
            return;
        checkItem(counter, residue.a, idx, a);
    }
}
template <typename Traits>
cudaError_t launch(uint32_t *__restrict__ counter, SchurVec<Traits, true> residue, typename HessianBase<Traits>::EcBlock c,
                   typename HessianBase<Traits>::EaBlock a, cudaStream_t stream){
    constexpr uint32_t ctaSize = 128;
    const uint32_t nbCtaC = divUp(residue.nbCBlocks, ctaSize);
    const uint32_t nbCtaA = divUp(residue.nbABlocks, ctaSize);
    kernel<<<(nbCtaC+nbCtaA), ctaSize, 0, stream>>>(counter, residue, c, a);
    return cudaGetLastError();
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launch(uint32_t *__restrict__ counter, SchurVec<TRAITS, true> residue, typename HessianBase<TRAITS>::EcBlock c,\
                                typename HessianBase<TRAITS>::EaBlock a, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES
}

}//namespace pcg
}//namespace rba
