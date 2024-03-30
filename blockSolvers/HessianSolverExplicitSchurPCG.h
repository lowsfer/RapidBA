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
#include "SchurSolverPCG.h"
#include "SchurSolver.h"
#include "HessianSolver.h"
#include "../computeHessian.h"

namespace rba {
namespace solver {
using grouped::DeviceHessian;
using grouped::DeviceModel;
using grouped::DeviceSchurPreCompData;
using grouped::DeviceSchur;
using grouped::DevSchurVec;

template <typename Traits>
class HessianSolverExplicitSchurPCG : public HessianSolverWSchur<Traits>{
    using typename HessianSolverWSchur<Traits>::Threshold;
public:
    void setUp(const rba::grouped::DeviceHessian<Traits> *hessian, const HessianVec<Traits, false> &x,
               const rba::grouped::DeviceModel<Traits> *model, cudaStream_t stream) override;

    // every call of solve will re-compute schur
    bool solve(const Threshold &threshold, uint32_t maxIters, cudaStream_t stream) override;

protected:
    void computeSchur(cudaStream_t stream) const;

protected:
    std::unique_ptr<DeviceSchur<Traits>> devSchur;
    std::unique_ptr<DeviceSchurPreCompData> devSchurPreComp;

    std::unique_ptr<SchurSolver<Traits>> solver;
};

} // namespace solver
} // namespace rba

