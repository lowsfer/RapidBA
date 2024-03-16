//
// Created by yao on 2/12/18.
//

#pragma once
//fixme: move SchurVec out of namespace pcg
#include "solveSchurPCG.h"

namespace rba {
namespace solver {

template <typename Traits>
class SchurSolver {
public:
    enum class Type{
        PCG
    };
    struct Threshold {typename HessianBase<Traits>::EcBlock c; typename HessianBase<Traits>::EaBlock a;};
    // set up space for solving. A and x do not need to contain the data used for solving but should have correct structure.
    // It is allowed to call setUp() only once and solve() multiple times with different content, as long as the structure and memory for A, x and b are not changed.
    virtual void setUp(const rba::grouped::DeviceSchur<Traits> &devA, const pcg::SchurVec<Traits, false> &x, cudaStream_t stream) = 0;
    virtual bool solve(const Threshold &threshold, uint32_t maxIters, cudaStream_t stream) = 0;
    virtual ~SchurSolver() = default;
};

template <typename Traits>
std::unique_ptr<SchurSolver<Traits>> createSchurSolver(typename SchurSolver<Traits>::Type type);

}//namespace solver
}//namespace rba
