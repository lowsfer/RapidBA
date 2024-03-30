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
// Created by yao on 6/12/18.
//

#include "HessianSolver.h"
#include "../kernel.h"
#include "../GroupModelTypes.h"
#include "../GroupModel.h"
#include "solveHessian.h"
#include "HessianSolverExplicitSchurPCG.h"
#include "../computeHessian.h"
#include <boost/preprocessor/seq/for_each.hpp>

namespace rba {
namespace solver {

template <typename Traits>
std::unique_ptr<HessianSolver<Traits>> createHessianSolver(typename HessianSolver<Traits>::Type type)
{
    switch (type){
        case HessianSolver<Traits>::Type::explicitSchur:
        {
            return std::unique_ptr<HessianSolver<Traits>>{new HessianSolverExplicitSchurPCG<Traits>()};
        }
    }
    throw std::runtime_error("fatal error");
}

template <typename Traits>
void HessianSolverWSchur<Traits>::clear() {
    devHessian = nullptr;
    hessianX.clear();
    devModel = nullptr;
    if (sparseIdxW2idxLocalOb != nullptr)
        sparseIdxW2idxLocalOb->clear();
}

template <typename Traits>
void HessianSolverWSchur<Traits>::setUp(const rba::grouped::DeviceHessian<Traits> *hessian, const HessianVec<Traits, false> &x,
                                        const rba::grouped::DeviceModel<Traits> *model, cudaStream_t stream) {
    devHessian = hessian;
    hessianX = x;
    devModel = model;
    if (devInvVBlocksCapacity < hessian->V.size()) {
        devInvVBlocks.reset(deviceAlloc<VSymBlock>(hessian->V.size()).release());
        devInvVBlocksCapacity = hessian->V.size();
    }
    if (sparseIdxW2idxLocalOb == nullptr)
        sparseIdxW2idxLocalOb = std::make_shared<DevVecVec<uint16_t>>();
    grouped::toDevice_sparseIdxW2idxLocalOb(*sparseIdxW2idxLocalOb, grouped::make_sparseIdxW2idxLocalOb(*model, *hessian));
    checkCudaErrors(migrateToDevice(getCudaDevice(), stream));
}

template <typename Traits>
void HessianSolverWSchur<Traits>::computeInvV(cudaStream_t stream) const {
    if (devHessian->getKernelArgsConst().nbVarPts != 0) {
        checkCudaErrors(launchCudaComputeInvV<Traits>(devInvVBlocks.get(), devHessian->getKernelArgsConst().V,
                                                      devHessian->getKernelArgsConst().nbVarPts, stream));
    }
}

template <typename Traits>
void HessianSolverWSchur<Traits>::solveB(cudaStream_t stream) {
    const auto hessian = devHessian->getKernelArgsConst();
    if (hessian.nbVarPts == 0)
        return;
    const pcg::SchurVec<Traits, true> deltaCA{hessianX.c, hessianX.nbCBlocks, hessianX.a, hessianX.nbABlocks};
    checkCudaErrors(cudaMemcpyAsync(hessianX.b, hessian.Eb, sizeof(hessianX.b[0])*hessian.nbVarPts, cudaMemcpyDeviceToDevice, stream));
    {
        // these two can run in parallel if use cuda graph
        assert(hessian.nbVarIntri == deltaCA.nbCBlocks);
        checkCudaErrors(solve_deltaB::launchCudaComputeSTDeltaC<Traits>(hessianX.b, hessian.S, deltaCA.c, stream));
        assert(deltaCA.nbABlocks == hessian.nbVarCaps);
        checkCudaErrors(solve_deltaB::launchCudaComputeWTDeltaA(hessianX.b, devModel->getKernelArgs(),
                sparseIdxW2idxLocalOb->getConst(), deltaCA.a, hessian.W, stream));
    }
    checkCudaErrors(solve_deltaB::launchCudaComputeMVInplace<Traits>(hessianX.b, devInvVBlocks.get(), hessian.nbVarPts, stream));
}

#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template std::unique_ptr<HessianSolver<TRAITS>> createHessianSolver(typename HessianSolver<TRAITS>::Type type);\
    template class HessianSolverWSchur<TRAITS>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES
}
}