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

#pragma once
#include "computeHessian.h"

namespace rba {

/** structure of Schur equation
 * | M,  Q|   | deltaC |   | Ec |
 * | Qt, U| * | deltaA | = | Ea |
 */
template<typename Traits, bool isConst = false>
struct Schur {
#define USING_BASE_TYPE(type) using type = ConstSelectType<typename HessianBase<Traits>::type, isConst>
    USING_BASE_TYPE(MBlock);
    USING_BASE_TYPE(MSymBlock);
    USING_BASE_TYPE(EcBlock);
    USING_BASE_TYPE(QBlock);
    USING_BASE_TYPE(UBlock);
    USING_BASE_TYPE(USymBlock);
    USING_BASE_TYPE(EaBlock);
#undef USING_BASE_TYPE

    //@todo: Instead of find out all non-zero elements in Schur, one alternative method to decide sparsity pattern of M and U is to use successfully solved image pairs, rather than relying on commonly seen points. This excludes indirectly connected captures/intrinsics pairs (usually with fewer commonly seen points) and makes it more sparse. Another is to directly filter out pairs with few common points (may break the graph?).

    IdxMBlock nbVarIntri;
    CSR<MBlock> upperM; //nbRows = nbVarIntri. Diagonal is not stored here. Last row is always empty
    MSymBlock *__restrict__ diagM;
    EcBlock *__restrict__ Ec;

    IdxUBlock nbVarCaps;
    CSR<UBlock> upperU; //nbRows = nbVarCaps. Diagonal is not stored here.  Last row is always empty
    USymBlock *__restrict__ diagU;
    EaBlock *__restrict__ Ea;

    CSR<QBlock> Q;
};

namespace SchurUpperMComputer {
struct PreComp {
    // same structure as Schur::upperM::data. LUT to find out which row it is in (similar to Schur::M::idxCol but for rows)
    const IdxMBlock *__restrict__ rowTableUpperM;

    struct alignas(8) PairSS {
        //both are sparseIdxS
        uint32_t a;
        uint32_t b;
    };
    // @fixme info: Note that n-th row is [roundUp(pairSS.rows[n], 128/sizeof(PairSS)), pairSS.rows[n+1]). Note that rows for diagM are empty as they are trivial.
    // nbRows = Shur::upperM::data[Schur::upperM::getRowEnd(Schur::upperM::nbRows)], i.e. number of non-zero elements
    VecVec<const PairSS> pairSS; // for Schur::upperM
};
}

template <typename Traits>
cudaError_t launchCudaComputeSchurUpperM(
        Hessian<Traits, true> hessian, const typename HessianBase<Traits>::VSymBlock *__restrict__ invV,
        Schur<Traits, false> schur, SchurUpperMComputer::PreComp preComp,
        cudaStream_t stream);

template <typename Traits>
cudaError_t launchCudaComputeSchurDiagM(
        Hessian<Traits, true> hessian, const typename HessianBase<Traits>::VSymBlock *__restrict__ invV,
        Schur<Traits, false> schur, cudaStream_t stream);

namespace SchurUpperUComputer {
// Two cameras (A and B) observes the same point (idxPt)
struct alignas(4) ObPair {
    uint16_t idxLocalObA;
    uint16_t idxLocalObB;
};
struct PreComp{
    const IdxUBlock *__restrict__ rowTableUpperU;
    VecVec<const SchurUpperUComputer::ObPair> pairs;
};
}
template <typename Traits>
cudaError_t launchCudaComputeSchurUpperU(
        Model<Traits, true> model,
        VecVec<const SchurUpperUComputer::ObPair> taskList, // nbRows = nbUpperUItems
        Hessian<Traits, true> hessian, //hessian is not needed. Put it here just for validation with assertions
        const typename HessianBase<Traits>::VSymBlock *__restrict__ invVList,
        Schur<Traits, false> schur,
        const IdxUBlock *__restrict__ rowTableUpperU, // same structure as schur.upperU.idxCols
        cudaStream_t stream);

template <typename Traits>
cudaError launchCudaComputeSchurDiagU(
        Model<Traits, true> model,
        VecVec<const uint16_t> sparseIdxW2idxLocalOb, // list of local ob index for variable points. nbRows = nbUpperUItems
        Hessian<Traits, true> hessian,
        const typename HessianBase<Traits>::VSymBlock *__restrict__ invVList,
        Schur<Traits, false> schur,
        cudaStream_t stream);

namespace SchurQComputer{
struct PreComp{
    VecVec<const uint32_t> taskListSparseIdxS; // nbRows = nbQItems
    VecVec<const uint16_t> taskListIdxLocalOb; //same shape as qPairSparseIdxS
    const IdxMBlock* rowTableQ; // idxRow with the same structure as schur.Q.idxCol
};
}
template <typename Traits>
cudaError_t launchCudaComputeSchurQ(
        Model<Traits, true> model,
        VecVec<const uint32_t> taskListSparseIdxS, // nbRows = nbQItems
        VecVec<const uint16_t> taskListIdxLocalOb, //same shape as qPairSparseIdxS
        const IdxMBlock* rowTableQ, // idxRow with the same structure as schur.Q.idxCol
        Hessian<Traits, true> hessian,
        const typename HessianBase<Traits>::VSymBlock* __restrict__ invVList,
        Schur<Traits, false> schur,
        cudaStream_t stream);

template <typename Traits>
cudaError_t launchCudaComputeInvV(typename HessianBase<Traits>::VSymBlock* __restrict__ dst, const typename HessianBase<Traits>::VSymBlock* __restrict__ src, uint32_t nbVBlocks, cudaStream_t stream);

struct SchurPreComp{
    SchurUpperMComputer::PreComp upperM;
    SchurUpperUComputer::PreComp upperU;
    VecVec<const uint16_t> sparseIdxW2idxLocalOb;//for diagU
    SchurQComputer::PreComp Q;
};

// invVList is not part of precomp because precomp is constant across the whole bundle adjustment, while invVlist needs to be update in every L-M step
template <typename Traits>
cudaError_t launchCudaComputeSchur(Model<Traits, true> model, Hessian<Traits, true> hessian, const typename HessianBase<Traits>::VSymBlock *invVList,
                                   Schur<Traits, false> schur,
                                   SchurPreComp precomp, cudaStream_t stream);



}//namespace rba


