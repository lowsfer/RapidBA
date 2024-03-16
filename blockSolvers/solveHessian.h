//
// Created by yao on 7/12/18.
//

#pragma once
#include "../computeHessian.h"
namespace rba {

namespace solve_deltaB {
// functions used for computing deltaB
template <typename Traits>
cudaError_t launchCudaComputeSTDeltaC(
        typename HessianBase<Traits>::EbBlock *acc, const CSR<const typename HessianBase<Traits>::SBlock> &hessianS,
        const typename HessianBase<Traits>::EcBlock *deltaCBlocks, cudaStream_t stream);

template <typename Traits>
cudaError_t launchCudaComputeWTDeltaA(
        typename HessianBase<Traits>::EbBlock *acc, const Model<Traits, true> &model,
        const VecVec<const uint16_t> &sparseIdxW2idxLocalOb, // list of local ob index for variable points. nbRows = nbVarCaps
        const typename HessianBase<Traits>::EaBlock *deltaABlocks,
        const CSR<const typename HessianBase<Traits>::WBlock> &hessianW,
        cudaStream_t stream);

template <typename Traits>
cudaError launchCudaComputeMVInplace(typename HessianBase<Traits>::EbBlock *vectors, const typename HessianBase<Traits>::VSymBlock *symMatrices,
                              uint32_t nbBlocks, cudaStream_t stream);
}// namespace deltaB
}// namespace rba