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
// Created by yao on 8/11/18.
//

#pragma once
#include "kernel.h"
#include <cassert>
#include "csr.h"
#include "utils_host.h"
namespace rba {

/** structure of Hessian equation. E is g in SBA paper
 * | M,  Q,  S |   | deltaC |   | Ec |
 * | Qt, U,  W | * | deltaA | = | Ea |
 * | St, Wt, V |   | deltaB |   | Eb |
 */
template <typename Traits>
struct HessianBase{
    RBA_IMPORT_TRAITS(Traits);
    using MBlock = kmat<epf, CamIntr::DoF, CamIntr::DoF>;
    using MSymBlock = symkmat<epf, CamIntr::DoF>;
    using EcBlock = kmat<epf, CamIntr::DoF>;
    using UBlock = kmat<hpf, Capture::DoF, Capture::DoF>;
    using USymBlock = symkmat<hpf, Capture::DoF>;
    using EaBlock = kmat<hpf, Capture::DoF>;
    using VBlock = kmat<lpf, Point<lpf>::DoF, Point<lpf>::DoF>;
    using VSymBlock = symkmat<lpf, Point<lpf>::DoF>;
    using EbBlock = kmat<lpf, Point<lpf>::DoF>;
    using QBlock = kmat<hpf, CamIntr::DoF, Capture::DoF>;
    using SBlock = kmat<lpf, CamIntr::DoF, Point<lpf>::DoF>;
    using WBlock = kmat<lpf, Capture::DoF, Point<lpf>::DoF>;
};

template<typename Traits, bool isConst = false>
struct Hessian {
#define USING_BASE_TYPE(type) using type = ConstSelectType<typename HessianBase<Traits>::type, isConst>
    USING_BASE_TYPE(MBlock);
    USING_BASE_TYPE(MSymBlock);
    USING_BASE_TYPE(EcBlock);
    USING_BASE_TYPE(UBlock);
    USING_BASE_TYPE(USymBlock);
    USING_BASE_TYPE(EaBlock);
    USING_BASE_TYPE(VBlock);
    USING_BASE_TYPE(VSymBlock);
    USING_BASE_TYPE(EbBlock);
    USING_BASE_TYPE(QBlock);
    USING_BASE_TYPE(SBlock);
    USING_BASE_TYPE(WBlock);
#undef USING_BASE_TYPE

    uint32_t nbVarIntri;
    MSymBlock *__restrict__ M;
    EcBlock *__restrict__ Ec;

    uint32_t nbVarCaps;
    USymBlock *__restrict__ U;
    EaBlock *__restrict__ Ea;

    uint32_t nbVarPts;
    VSymBlock *__restrict__ V;
    EbBlock *__restrict__ Eb;

//    CSR<ConstSelectType<QBlock, isConst>> Q;
    // @info: each column of sparse matrix Q has at most one QBlock. We make use of a special format (a dense row) relying on this feature to further optimize storage.
    // Note that we don't support single-extrinsics multi-intrinsics, i.e. stereo cameras or multi-view cameras rigs. If we change this, W also need to be modified with atomicAdd()
    struct {
        // length = nbVarCaps for both pointers
        QBlock *__restrict__ blocks;//a dense row
        const uint32_t *__restrict__ row;// idxMBlock. idxVarFixed if the intrinsics are fixed.
    } Q;

    CSR<SBlock> S;

    CSR<WBlock> W;
};

template <bool isConst>
struct HessianPreCompData{
    // shape = (nbVarCaps, nbObsInCap). sparseIdxS.idxCols = nullptr and sparseIdxV.idxCols = nullptr. These are LUT, not really CSR
    uint32_t nbVarCaps;
    uint32_t nbInvolvedCaps;
    const uint32_t* __restrict__ rows;// length = nbInvolvedCaps+1
    const uint32_t* __restrict__ sparseIdxS; //length = rows[nbInvolvedCaps]
    const uint32_t* __restrict__ sparseIdxW; //length = rows[nbVarCaps]
};

#if __CUDACC_VER_MAJOR__ >= 10
template <typename Traits>
UniqueCudaGraph cudaComputeHessian(Model<Traits, true> model, HessianPreCompData<true> precomp, Hessian<Traits, false> hessian);
#endif
template <typename Traits>
cudaError_t launchCudaComputeErrSqrNorm(const Model<Traits, true>& model, double *errSqrNorm, cudaStream_t stream);
template <typename Traits>
cudaError_t launchCudaComputeHessian(const Model<Traits, true>& model, double* errSqrNorm, const Hessian<Traits, false>& hessian, uint32_t nbSBlocks, const HessianPreCompData<true>& precomp, cudaStream_t stream = nullptr);

// @todo: this is not yet implemented
// remove one position DoF of one extrinsics to fix the scale of the system. This is needed when only 6 DoF (i.e. one Capture) is fixed in the system. The system needs 7 DoF fixed to be determined
template <typename Traits>
cudaError_t launchCudaRemoveOneDoF(Hessian<Traits, false> hessian, IdxUBlock idxU, int idxDim/* 0, 1 or 2 */, cudaStream_t stream);

// @todo: this is not yet implemented
// add constraint that ||delta|| should be minimized, in case the problem is under-determined. See https://en.wikipedia.org/wiki/Matrix_regularization
template <typename Traits>
cudaError_t launchCudaRegularizeHessian(Hessian<Traits, false> hessian,float lambda, cudaStream_t stream);

template<typename Traits, bool isConst>
struct HessianVec {
#define USING_BASE_TYPE(type) using type = ConstSelectType<typename HessianBase<Traits>::type, isConst>
    USING_BASE_TYPE(EcBlock);
    USING_BASE_TYPE(EaBlock);
    USING_BASE_TYPE(EbBlock);
    EcBlock *__restrict__ c;
    uint32_t nbCBlocks;
    EaBlock *__restrict__ a;
    uint32_t nbABlocks;
    EbBlock *__restrict__ b;
    uint32_t nbBBlocks;
#undef USING_BASE_TYPE

    void clear() {
        c = nullptr; nbCBlocks = 0;
        a = nullptr; nbABlocks = 0;
        b = nullptr; nbBBlocks = 0;
    }
    template<bool constResult = true, typename std::enable_if<!isConst && constResult, int>::type = 0>
    operator HessianVec<Traits, constResult>() const {
        return HessianVec<Traits, constResult>{c, nbCBlocks, a, nbABlocks, b, nbBBlocks};
    }
};
namespace dampLM {
template <typename Traits>
cudaError_t launchCudaBackupAndDamp(HessianVec<Traits, false> backup, Hessian<Traits, false> hessian, float damp, cudaStream_t stream);

template <typename Traits>
cudaError_t launchCudaDampFromBackup(HessianVec<Traits, true> backup, Hessian<Traits, false> hessian, float damp, cudaStream_t stream);

template <typename Traits>
cudaError_t launchCudaComputeExpectedErrSqrNormDelta(float* expectedErrSqrNormDelta, float damp,
        const HessianVec<Traits, true>& delta, const HessianVec<Traits, true>& diagBackup,
        const HessianVec<Traits, true>& g, // Ec, Ea and Eb of hessian
        cudaStream_t stream);
}
namespace updateModel {
// Don't move this to BackupData, otherwise you get circular dependency
template<typename Traits, bool isConst>
struct BackupData {
    RBA_IMPORT_TRAITS(Traits);
    ConstSelectType<CamIntr, isConst> *__restrict__ cameras;
    uint32_t nbVarIntri;
    ConstSelectType<Capture, isConst> *__restrict__ captures;
    uint32_t nbVarCap;
    ConstSelectType<Point<lpf>, isConst> *__restrict__ points;
    uint32_t nbVarPts;
};
//@fixme: need tests
template <typename Traits>
cudaError_t launchCudaUpdateModel(const BackupData<Traits, false>& backup, const Model<Traits, false>& model, const HessianVec<Traits, true>& delta, cudaStream_t stream);
//@fixme: need tests
template <typename Traits>
cudaError_t launchCudaRevertModel(const Model<Traits, false>& model, const BackupData<Traits, true>& backup, cudaStream_t stream);
}
} // namespace rba
