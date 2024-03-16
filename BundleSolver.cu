//
// Created by yao on 12/12/18.
//

#include "cuda_hint.cuh"
#include <cuda_runtime.h>
#include <limits>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <cuda_runtime.h>
#include "utils_kernel.h"
#include "utils_general.h"
#include "utils_host.h"
#include "computeHessian.h" // For HessianVec
#include <boost/preprocessor/seq/for_each.hpp>

namespace rba{

template <typename AccType>
struct AbsMaxOp
{
    __device__ inline
    static AccType initVal() { return 0; }

    template<typename ValType>
    __device__ inline
    static void reduce(AccType& acc, ValType val) { acc = std::max(acc, std::abs(AccType(val))); }

    __device__ inline
    static void combine(AccType& acc0, AccType acc1) { acc0 = std::max(acc0, acc1); }

    __device__ inline
    static void atomicCombine(AccType& acc0, AccType acc1, unsigned cas_yield_nanosec = 20) { atomicMax(&acc0, acc1); (void)cas_yield_nanosec;}
};

__device__ inline float atomicCAS(float *address, float compare, float val){
    const uint32_t result = ::atomicCAS(reinterpret_cast<uint32_t*>(address), reinterpret_copy<uint32_t>(compare), reinterpret_copy<uint32_t>(val));
    return reinterpret_copy<float>(result);
}

__device__ inline double atomicCAS(double *address, double compare, double val){
    using uint64_type = unsigned long long int;
    const uint64_type result = ::atomicCAS(reinterpret_cast<uint64_type*>(address), reinterpret_copy<uint64_type>(compare), reinterpret_copy<uint64_type>(val));
    return reinterpret_copy<double>(result);
}

template <typename T>
__device__ inline
void atomicCASMax(T* dst, T src, unsigned yield_nanosec = 0){
    T old = *dst;
    T assumed;
    bool firstIter = true;
    unused(yield_nanosec); unused(firstIter);
    do {
        assumed = old;
        const T result = std::max(old, src);
#if __CUDA_ARCH__ > 700
        if (!firstIter){
            __nanosleep(yield_nanosec);
            firstIter = false;
        }
#endif
        old = atomicCAS(dst, assumed, result);
    }while(old != assumed);
}

template <>
__device__ inline
void AbsMaxOp<float>::atomicCombine(float& acc0, float acc1, unsigned yield_nanosec) {
    atomicCASMax(&acc0, acc1, yield_nanosec);
}
template <>
__device__ inline
void AbsMaxOp<double>::atomicCombine(double& acc0, double acc1, unsigned yield_nanosec) {
    atomicCASMax(&acc0, acc1, yield_nanosec);
}

template <template <typename> class RedOp, typename AccType>
__global__ void kernel_initRedAcc(AccType* __restrict__ acc)
{
    *acc = RedOp<AccType>::initVal();
}
template <template <typename> class RedOp, typename AccType>
cudaError_t launchCudaInitRedAcc(AccType* acc, cudaStream_t stream)
{
    return launchKernel(kernel_initRedAcc<RedOp, AccType>, 1, 1, 0, stream, acc);
}

template <template <typename> class RedOp, typename AccType, typename ValType, uint32_t ctaSize = 128, uint32_t innerLoop = 8>
__global__ void kernel_reduce(AccType* __restrict__ acc, const ValType* __restrict__ data, uint32_t size)
{
    const uint32_t outerLoop = divUp(size, ctaSize * innerLoop * gridDim.x);
    const uint32_t ctaOffset = ctaSize*innerLoop * outerLoop * blockIdx.x;
    if (ctaOffset > size){
        return;
    }

    using Red = RedOp<AccType>;
    __shared__ AccType ctaAcc;
    if (threadIdx.x == 0){
        ctaAcc = Red::initVal();
    }
    __syncthreads();
    // thread reduction
    AccType thrdAcc[innerLoop];
    for (auto& v : thrdAcc) {
        v = Red::initVal();
    }
    for (uint32_t i = 0; i < outerLoop; i++)
    {
        const uint32_t offset = ctaOffset + innerLoop *ctaSize * i;
        if (offset >= size)
            break;
#pragma unroll
        for (uint32_t j = 0; j < innerLoop; j++)
        {
            const uint32_t idx = offset + ctaSize * j + threadIdx.x;
            //@todo: test performance against if (idx >= size) break;
            if (idx < size) {
                const ValType val = data[idx];
                Red::reduce(thrdAcc[j], val);
            }
        }
    }
    static_assert((innerLoop & (innerLoop - 1)) == 0, "fatal error");
    for (uint32_t mask = innerLoop; mask > 1u; mask /= 2){
        for (uint32_t i = 0; i < mask / 2; i++){
            Red::combine(thrdAcc[i], thrdAcc[i + mask/2]);//or (i^mask)
        }
    }
    //warp reduction
    assert(warp_size == warpSize);
    const int lane_id = int(threadIdx.x % warp_size);
    for (int mask = warp_size; mask != 1u; mask /= 2){
        const AccType other = __shfl_xor_sync(~0u, thrdAcc[0], mask/2);
        if (lane_id < mask) {
            Red::combine(thrdAcc[0], other);
        }
    }
    //CTA reduction
    if (lane_id == 0) {
        Red::atomicCombine(ctaAcc, thrdAcc[0], 20);
    }
    __syncthreads();
    //global reduction
    if (threadIdx.x == 0){
        Red::atomicCombine(*acc, ctaAcc, 200);
    }
}

template <typename AccType>
struct SquaredNormOp
{
    __device__ inline
    static AccType initVal() { return 0; }

    template<typename ValType>
    __device__ inline
    static void reduce(AccType& acc, ValType val) { acc += sqr(AccType(val)); }

    __device__ inline
    static void combine(AccType& acc0, AccType acc1) { acc0 += acc1; }

    __device__ inline
    static void atomicCombine(AccType& acc0, AccType acc1, unsigned cas_yield_nanosec = 20) { atomicAdd(&acc0, acc1); (void)cas_yield_nanosec;}
};

template <typename Traits, template <typename> class RedOp, typename AccType, uint32_t ctaSize = 128, uint32_t innerLoop = 8>
cudaError_t launchCudaReduce(AccType* acc, const HessianVec<Traits, true>& vec, cudaStream_t stream)
{
    checkEarlyReturn(launchCudaInitRedAcc<RedOp>(acc, stream));
    cudaError_t err = cudaSuccess;
    if (vec.nbCBlocks > 0) {
        static_assert(!isGroupModel<Traits>() || sizeof(vec.c[0][0]) * std::decay_t<decltype(vec.c[0])>::size() == sizeof(vec.c[0]), "fatal error");
        err = launchKernel(kernel_reduce<RedOp, AccType, typename Traits::epf, ctaSize, innerLoop>, 4, ctaSize, 0, stream,
                acc, vec.c[0].data(), vec.c[0].size() * vec.nbCBlocks);
        checkEarlyReturn(err);
    }
    if (vec.nbABlocks > 0) {
        static_assert(sizeof(vec.a[0][0]) * std::decay_t<decltype(vec.a[0])>::size() == sizeof(vec.a[0]), "fatal error");
        err = launchKernel(kernel_reduce<RedOp, AccType, typename Traits::hpf, ctaSize, innerLoop>, 16, ctaSize, 0, stream,
            acc, vec.a[0].data(), vec.a[0].size() * vec.nbABlocks);
        checkEarlyReturn(err);
    }
    if (vec.nbBBlocks) {
        static_assert(sizeof(vec.b[0][0]) * std::decay_t<decltype(vec.b[0])>::size() == sizeof(vec.b[0]), "fatal error");
        err = launchKernel(kernel_reduce<RedOp, AccType, typename Traits::lpf, ctaSize, innerLoop>, 128, ctaSize, 0, stream,
                acc, vec.b[0].data(), vec.b[0].size() * vec.nbBBlocks);
        checkEarlyReturn(err);
    }
    return cudaSuccess;
}

template <typename Traits>
cudaError_t launchCudaGetAbsMax(float* absMaxVal, const HessianVec<Traits, true>& vec, cudaStream_t stream)
{
    return launchCudaReduce<Traits, AbsMaxOp, float, 128, 8>(absMaxVal, vec, stream);
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchCudaGetAbsMax(float* absMaxVal, const HessianVec<TRAITS, true>& vec, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template <typename Traits>
cudaError_t launchCudaGetSquaredNorm(float* squaredNorm, const HessianVec<Traits, true>& vec, cudaStream_t stream)
{
    return launchCudaReduce<Traits, SquaredNormOp, float, 128, 8>(squaredNorm, vec, stream);
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template cudaError_t launchCudaGetSquaredNorm(float* squaredNorm, const HessianVec<TRAITS, true>& vec, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template <typename AccType>
struct ModelVarSquaredNormOp
{
    __device__ inline
    static AccType initVal() { return 0; }

    template<typename VarType>
    __device__ inline
    static void reduce(AccType& acc, VarType var) { acc += AccType(var.squaredNorm()); }

    __device__ inline
    static void combine(AccType& acc0, AccType acc1) { acc0 += acc1; }

    __device__ inline
    static void atomicCombine(AccType& acc0, AccType acc1, unsigned cas_yield_nanosec = 20) { atomicAdd(&acc0, acc1); (void)cas_yield_nanosec;}
};

template <typename Traits>
cudaError_t launchCudaGetModelVarSquaredNorm(float* varSquaredNorm, const updateModel::BackupData<Traits, true>& backup, cudaStream_t stream)
{
    RBA_IMPORT_TRAITS(Traits);
    using AccType = float;
    constexpr uint32_t ctaSize = 128;
    constexpr uint32_t innerLoop = 8;
    AccType* acc = varSquaredNorm;
    struct VecType{
        const CamIntr* __restrict__ c;
        uint32_t nbCBlocks;
        const Capture* __restrict__ a;
        uint32_t nbABlocks;
        const Point<lpf>* __restrict__ b;
        uint32_t nbBBlocks;
    };
    const VecType vec{
            backup.cameras, backup.nbVarIntri,
            backup.captures, backup.nbVarCap,
            backup.points, backup.nbVarPts
    };

    checkEarlyReturn(launchCudaInitRedAcc<ModelVarSquaredNormOp>(acc, stream));

    if (vec.nbCBlocks > 0) {
        checkEarlyReturn(launchKernel(kernel_reduce<ModelVarSquaredNormOp, AccType, typename Traits::CamIntr, ctaSize, innerLoop>, 4, ctaSize, 0, stream,
                     acc, vec.c, vec.nbCBlocks));
    }
    if (vec.nbABlocks > 0) {
        checkEarlyReturn(launchKernel(kernel_reduce<ModelVarSquaredNormOp, AccType, Capture, ctaSize, innerLoop>, 16, ctaSize, 0, stream,
                     acc, vec.a, vec.nbABlocks));
    }
    if (vec.nbBBlocks) {
        checkEarlyReturn(launchKernel(kernel_reduce<ModelVarSquaredNormOp, AccType, rba::Point<lpf>, ctaSize, innerLoop>, 128, ctaSize, 0, stream,
                     acc, vec.b, vec.nbBBlocks));;
    }
    return cudaSuccess;
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
template cudaError_t launchCudaGetModelVarSquaredNorm(float* varSquaredNorm, const updateModel::BackupData<TRAITS, true>& backup, cudaStream_t stream);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES
} // namespace rba
