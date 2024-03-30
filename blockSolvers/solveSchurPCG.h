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
// Created by yao on 1/12/18.
//

#pragma once
#include "../computeSchur.h"
#include "../computeHessian.h"
#include <type_traits>

namespace rba
{
namespace pcg {
template <typename Traits, bool isConst>
struct SchurVec {
#define USING_BASE_TYPE(type) using type = ConstSelectType<typename HessianBase<Traits>::type, isConst>
    USING_BASE_TYPE(EcBlock);
//    using EcBlock = ConstSelectType<typename HessianBase<Traits>::EcBlock, isConst>;
    USING_BASE_TYPE(EaBlock);
//    using EaBlock = ConstSelectType<typename HessianBase<Traits>::EaBlock, isConst>;
    EcBlock * __restrict__ c;
    uint32_t nbCBlocks;
    EaBlock * __restrict__ a;
    uint32_t nbABlocks;
#undef USING_BASE_TYPE
    template <bool constResult = true>
    operator typename std::enable_if<!isConst, SchurVec<Traits, constResult>>::type() const {
        return SchurVec<Traits, constResult>{c, nbCBlocks, a, nbABlocks};
    }
};

struct SchurMMVPreComp{
    //store index to Schur::upperM::data
    CSR<const uint32_t> lowerM;//nbRows == Schur::nbVarIntri. First row is always empty
    CSR<const uint32_t> lowerU;//nbRows == Schur::nbVarCap. First row is always empty
    CSR<const uint32_t> lowerQ;
};

template <typename Traits, bool isConst>
struct SchurDiag {
#define USING_BASE_TYPE(type) using type = ConstSelectType<typename HessianBase<Traits>::type, isConst>
    USING_BASE_TYPE(MSymBlock);
    USING_BASE_TYPE(USymBlock);
    MSymBlock* __restrict__ M;
    uint32_t nbMBlocks;
    USymBlock* __restrict__ U;
    uint32_t nbUBlocks;
#undef USING_BASE_TYPE
};

// intiantiated for MSymBlock and USymBlock in solvePCG.cu
template <typename ftype, uint32_t nbRows>
cudaError_t launchCudaComputeInverse(symkmat<ftype, nbRows>* __restrict__ dst,
                                     const symkmat<ftype, nbRows>* __restrict__ src,
                                     uint32_t nbBlocks, cudaStream_t stream);

namespace SchurMMV {
// compute diag*vecA and optionally vecB*diag*vecA
template <typename Traits>
cudaError_t launchForDiag(
        const SchurVec<Traits, false> &dstVec, typename Traits::epf *dstScalar,
        const SchurDiag<Traits, true> &diag, const SchurVec<Traits, true> &vecA, const SchurVec<Traits, true> &vecB, cudaStream_t stream);
template <typename Traits>
cudaError_t launchForUpperM(typename HessianBase<Traits>::EcBlock *dst, const CSR<const typename HessianBase<Traits>::MBlock> &upperM,
                            const typename HessianBase<Traits>::EcBlock *Ec, cudaStream_t stream);
template <typename Traits>
cudaError_t launchForUpperU(typename HessianBase<Traits>::EaBlock *dst, const CSR<const typename HessianBase<Traits>::UBlock> &upperU,
                            const typename HessianBase<Traits>::EaBlock *Ea, cudaStream_t stream);

template <typename Traits>
cudaError_t launchForLowerM(typename HessianBase<Traits>::EcBlock *dst, const CSR<const uint32_t> &lowerM,
                            const CSR<const typename HessianBase<Traits>::MBlock> &upperM,
                            const typename HessianBase<Traits>::EcBlock *Ec, cudaStream_t stream);
template <typename Traits>
cudaError_t launchForLowerU(typename HessianBase<Traits>::EaBlock *dst, const CSR<const uint32_t> &lowerU,
                            const CSR<const typename HessianBase<Traits>::UBlock> &upperU,
                            const typename HessianBase<Traits>::EaBlock *Ea, cudaStream_t stream);
template <typename Traits>
cudaError_t launchForUpperQ(
        typename HessianBase<Traits>::EcBlock *__restrict__ dst, CSR<const typename HessianBase<Traits>::QBlock> blocks,
        const typename HessianBase<Traits>::EaBlock *__restrict__ vectors,
        cudaStream_t stream);
template <typename Traits>
cudaError_t launchForLowerQ(
        typename HessianBase<Traits>::EaBlock *__restrict__ dst, CSR<const uint32_t> blocks,
        CSR<const typename HessianBase<Traits>::QBlock> upper,
        const typename HessianBase<Traits>::EcBlock *__restrict__ vectors,
        cudaStream_t stream);
}

namespace SchurDotProd{
template <typename Traits>
cudaError_t launch(typename Traits::epf* __restrict__ acc,
                   const SchurVec<Traits, true>& vecA, const SchurVec<Traits, true>& vecB,
                   cudaStream_t stream);
}

namespace SchurVecAdd{
template <typename Traits>
cudaError_t launch(const SchurVec<Traits, false>& dst, const typename Traits::epf* alpha, bool negativeAlpha,
                   const SchurVec<Traits, true>& vecA, const SchurVec<Traits, true>& vecB, cudaStream_t stream);
}

template <typename ftype>
cudaError_t launchUpdateBeta(ftype& beta, ftype(&zTr)[2], cudaStream_t stream);

template <typename ftype>
cudaError_t launchUpdateAlpha(ftype& alpha, const ftype& zTr, const ftype& pTAp, cudaStream_t stream);

namespace CheckThreshold{
template <typename Traits>
cudaError_t launch(uint32_t *__restrict__ counter, SchurVec<Traits, true> residue, typename HessianBase<Traits>::EcBlock c,
        typename HessianBase<Traits>::EaBlock a, cudaStream_t stream);
}
}//namespace pcg
}//namespace rba
