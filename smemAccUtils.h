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
// Created by yao on 9/20/19.
//

#pragma once
#include "cuda_hint.cuh"
#include <cstdint>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
#include "kmat.h"
#include "utils_kernel.h"
namespace cg = cooperative_groups;

namespace rba{

namespace smemAccUtils {
constexpr unsigned getSmemLineWidth(){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    return 64u;
#else
    return 128u;
#endif
}

template<typename T>
static constexpr uint32_t getAccVecSize() { return getSmemLineWidth() / (sizeof(T)); }

template<typename T>
struct AccVecType : kmat<T, getAccVecSize<T>()> {
    template <size_t grpSize> __device__ __forceinline__
    T init(const cg::thread_block_tile<grpSize> &g) const {
        const uint32_t idx = get_lane_id();
        if (AccVecType<T>::size()) {
            (*this)[idx] = T(0);
        }
    }

    __device__ __forceinline__
    void atomicRed(T val) {
        const uint32_t idx = get_lane_id() % AccVecType<T>::size();
        atomicAdd(&(*this)[idx], val);
    }

    template <size_t grpSize> __device__ __forceinline__
    T thrdGrpSum(const cg::thread_block_tile<grpSize> &g) const {
        T sum = (*this)[g.thread_rank()];
        for (unsigned i = g.size() / 2; i != 0; i /= 2) {
            const T sum_other = g.shfl_xor(sum, i);
            sum += sum_other;
        }
        return sum;
    }
};

//! if dstRequireAtomicAdd == false, we use assign instead atomicAdd
template<size_t nbElems, typename DstScalar, typename SrcScalar, bool dstRequiresAtomicAdd = true>
__device__ __forceinline__
static void ctaRed(DstScalar *dst, const AccVecType<SrcScalar> *src) {
    const auto g = cg::tiled_partition<AccVecType<SrcScalar>::size()>(cg::this_thread_block());
    const unsigned nbGrps = cg::this_thread_block().size() / g.size();
    const unsigned idxGrp = cg::this_thread_block().thread_rank() / g.size();
    for (unsigned i = idxGrp; i < nbElems; i += nbGrps) {
        const SrcScalar sum = src[i].thrdGrpSum<AccVecType<SrcScalar>::size()>(g);
        if (g.thread_rank() == 0) {
            if (dstRequiresAtomicAdd) {
                atomicAdd(&dst[i], sum);
            } else {
                dst[i] = sum;
            }
        }
    }
}
}//namespace smemAccUtils
}
