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