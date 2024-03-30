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

#include "SchurSolverPCG.h"
#include "HessianSolverExplicitSchurPCG.h"
#include "../computeHessian.h"
#include <boost/preprocessor/seq/for_each.hpp>

namespace rba{
namespace solver{

template <typename Traits>
void HessianSolverExplicitSchurPCG<Traits>::setUp(const rba::grouped::DeviceHessian<Traits> *hessian,
                                                  const HessianVec<Traits, false> &x,
                                                  const rba::grouped::DeviceModel<Traits> *model, cudaStream_t stream) {
    HessianSolverWSchur<Traits>::setUp(hessian, x, model, stream);

    devSchur = std::make_unique<DeviceSchur<Traits>>();
    devSchurPreComp = std::make_unique<DeviceSchurPreCompData>();
    devSchur->init(*this->devModel, *this->devHessian, devSchurPreComp.get(), this->sparseIdxW2idxLocalOb);
    if (solver == nullptr)
        solver = std::make_unique<SchurSolverPCG<Traits>>();
    solver->setUp(*devSchur, pcg::SchurVec<Traits, false>{x.c, x.nbCBlocks, x.a, x.nbABlocks}, stream);
    {
        const int device = getCudaDevice();
        checkCudaErrors(devSchur->migrateToDevice(device, stream));
        checkCudaErrors(devSchurPreComp->migrateToDevice(device, stream));
    }
}

template <typename Traits>
bool HessianSolverExplicitSchurPCG<Traits>::solve(const Threshold &threshold, uint32_t maxIters,
                                   cudaStream_t stream) {
    this->computeInvV(stream);
    computeSchur(stream);
    const bool converged = solver->solve({threshold.c, threshold.a}, maxIters, stream);
    this->solveB(stream);
    return converged;
}

template <typename Traits>
void HessianSolverExplicitSchurPCG<Traits>::computeSchur(cudaStream_t stream) const {
    checkCudaErrors(rba::launchCudaComputeSchur(this->devModel->getKernelArgs(), this->devHessian->getKernelArgsConst(),
                                                this->devInvVBlocks.get(), devSchur->getKernelArgs(),
                                                devSchurPreComp->toParams(), stream));
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template class HessianSolverExplicitSchurPCG<TRAITS>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES
} // namespace solver
} // namespace rba
