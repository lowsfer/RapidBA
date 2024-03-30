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
