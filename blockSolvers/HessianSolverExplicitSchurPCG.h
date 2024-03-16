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

