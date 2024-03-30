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
// Created by yao on 15/09/18.
//

#pragma once
#include "cuda_hint.cuh"
#include "kmat.h"
#include <cstdint>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "utils_general.h"
#include "ldg.cuh"

#define USE_PTX 0

template <typename T>
__device__ __host__ __forceinline__
void crossAdd(T(&dst)[3], const T(&u)[3], const T(&v)[3], const T(&acc)[3] = {0, 0, 0}){
    for (int i = 0; i < 3; i++){
        dst[i] = acc[i] + u[(i+1)%3] * v[(i+2)%3] - u[(i+2)%3] * v[(i+1)%3];
    }
}

template <typename T, std::size_t Length>
__device__ __host__ __forceinline__
T dotAdd(const T(&u)[Length], const T(&v)[Length], T acc = 0){
    T dst = acc;
    for (int i = 0; i < Length; i++){
        dst += u[i] * v[i];
    }
    return dst;
}

__device__ __host__ __forceinline__ float fast_rcp(float x){
#if defined(__CUDACC__) && USE_PTX
    float result;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
#else
    return 1 / x;
#endif
}

__device__ __host__ __forceinline__ float fast_sqrt(float x){
#if defined(__CUDACC__) && USE_PTX
    float result;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
#else
    return std::sqrt(x);
#endif
}

__device__ __host__ __forceinline__ float fast_rsqrt(float x){
#if defined(__CUDACC__)
#if USE_PTX
    float result;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
#else
    return 1/std::sqrt(x);
#endif
#else
    return 1/std::sqrt(x);
#endif
}

//(u + v + u.cross(v)) / (1 - u.dot(v));
template <typename T>
__device__ __host__ __forceinline__
void gvec_mul(T(&dst)[3], const T(&u)[3], const T(&v)[3]){
    for (int i = 0; i < 3; i++)
        dst[i] = u[i] + v[i];
    crossAdd(dst, u, v, dst);
    const T scale = -fast_rcp(dotAdd(u,v, T(-1)));
    for (int i = 0; i < 3; i++)
        dst[i] *= scale;
}

template <typename T>
__device__ __host__ __forceinline__
T sqr(T x){return x*x;}

template <typename T>
__device__ __host__ __forceinline__
T cubic(T x){return x*x*x;}

template <std::size_t Length>
__device__ __host__ __forceinline__
kmat<float, Length> normalized(const kmat<float, Length>& src)
{
    float sqrNorm = src.sqrNorm();
    float factor = fast_rsqrt(sqrNorm);
    return src * factor;
}

// https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
template <typename T>
__device__ __host__ __forceinline__
kmat<T, 3, 3> quat2mat(const kmat<T, 4, 1>& q){
    assert(abs(q.sqrNorm() - 1.f) < 1E-4f);
    constexpr float sqrt2 = 1.4142135623730951f;
    const T w = q[0]*sqrt2, x = q[1]*sqrt2, y = q[2]*sqrt2, z = q[3]*sqrt2;//[w,x,y,z]*sqrt2
    return {{
        1-y*y-z*z, x*y-z*w, x*z+y*w,
        x*y+z*w, 1-x*x-z*z, y*z-x*w,
        x*z-y*w, y*z+x*w, 1-x*x-y*y
    }};
}

// https://en.wikipedia.org/wiki/Rotation_matrix#Skew_parameters_via_Cayley's_formula
template<typename T>
__device__ __host__ inline
kmat<T, 3, 3> gvec2mat(const kmat<T, 3, 1>& gvec)
{
    const T x = gvec[0];
    const T y = gvec[1];
    const T z = gvec[2];
    const T factor = 2 * fast_rcp(x*x + y*y + z*z + 1);
    kmat<T, 3, 3> R = {{
        x*x + 1,  x*y - z, x*z + y,
        x*y + z,  y*y + 1,  y*z - x,
        x*z - y,  y*z + x, z*z + 1
    }};
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            R(i, j) = R(i,j) * factor - (i==j ? 1 : 0);
        }
    }
    return R;
}

// It is said this method has an issue here http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/paul.htm
// but that example does not look like a valid rotation matrix
template <typename T>
__device__ __host__ inline
kmat<T, 4, 1> mat2quat_impl0(const kmat<T,3,3>& R)
{
    T w,x,y,z;

    const T tx = R(0,0) * 0.25f, ty = R(1,1) * 0.25f, tz = R(2,2) * 0.25f;
    w = fast_sqrt(std::max(T(0.f), 0.25f + tx + ty + tz));
    x = fast_sqrt(std::max(T(0.f), 0.25f + tx - (ty + tz)));
    y = fast_sqrt(std::max(T(0.f), 0.25f + ty - (tx + tz)));
    z = fast_sqrt(std::max(T(0.f), 0.25f + tz - (tx + ty)));
    x = std::copysign(x, R(2,1) - R(1,2));
    y = std::copysign(y, R(0,2) - R(2,0));
    z = std::copysign(z, R(1,0) - R(0,1));
    return {w,x,y,z};
}

// http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
template <typename T>
__device__ __host__ inline
kmat<T, 4, 1> mat2quat_impl1(const kmat<T,3,3>& R)
{
    T w,x,y,z;
    T trace = R(0,0) + R(1,1) + R(2,2);
    if( trace > 0 ) {
        T s = 0.5f / fast_sqrt(trace+ 1.0f);
        w = 0.25f / s;
        x = ( R(2,1) - R(1,2) ) * s;
        y = ( R(0,2) - R(2,0) ) * s;
        z = ( R(1,0) - R(0,1) ) * s;
    } else {
        if ( R(0,0) > R(1,1) && R(0,0) > R(2,2) ) {
            const T s = 2.0f * fast_sqrt( 1.0f + R(0,0) - R(1,1) - R(2,2));
            const T s_inv = fast_rcp(s);
            w = (R(2,1) - R(1,2) ) * s_inv;
            x = 0.25f * s;
            y = (R(0,1) + R(1,0) ) * s_inv;
            z = (R(0,2) + R(2,0) ) * s_inv;
        } else if (R(1,1) > R(2,2)) {
            T s = 2.0f * fast_sqrt( 1.0f + R(1,1) - R(0,0) - R(2,2));
            const T s_inv = fast_rcp(s);
            w = (R(0,2) - R(2,0) ) * s_inv;
            x = (R(0,1) + R(1,0) ) * s_inv;
            y = 0.25f * s;
            z = (R(1,2) + R(2,1) ) * s_inv;
        } else {
            T s = 2.0f * fast_sqrt( 1.0f + R(2,2) - R(0,0) - R(1,1) );
            const T s_inv = fast_rcp(s);
            w = (R(1,0) - R(0,1) ) * s_inv;
            x = (R(0,2) + R(2,0) ) * s_inv;
            y = (R(1,2) + R(2,1) ) * s_inv;
            z = 0.25f * s;
        }
    }
    return {w,x,y,z};
}

template <typename T>
__device__ __host__ inline
kmat<T, 4, 1> mat2quat(const kmat<T,3,3>& R){
    return mat2quat_impl0(R);
}

__device__ __forceinline__ uint32_t get_lane_id(){
    uint32_t result;
    asm("mov.u32 %0, %%laneid;" : "=r"(result));
    return result;
}

constexpr unsigned warp_size = 32;
#ifdef __CUDACC__
// reference implementation of warpAtomicAdd() for tests
template<typename ValType>
__device__ inline ValType warpSumRef(ValType val){
    const auto g = cooperative_groups::coalesced_threads();
    ValType sum = 0;
    for (unsigned i = 0; i < g.size(); i++)
        sum += g.shfl(val, i);
    return sum;
}

// @fixme: need tests
// @fixme: move isFullWarp to first arg and set it explicitly
// This actually works only if isFullWarp == true.
template<bool isFullWarp, typename AccType, typename ValType>
__device__ __noinline__ void warpAtomicAdd(AccType* const acc, const ValType val){
    namespace cg = cooperative_groups;
#if 0
    //@fixme: It is said compiler can do warp aggregated atomic for many cases (https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/). Test if compiler can do it for this case.
    atomicAdd(acc, val);
#else
 	// no need for __syncwarp() here and the else branch later uses cg::coalesced_threads()
    const unsigned activemask = isFullWarp ? ~0u : __activemask();
    assert((__syncwarp(), __activemask() == activemask));
#ifndef NDEBUG
	if(isFullWarp) {
		__syncwarp();
	}
    const ValType sumRef = warpSumRef(val);//for test assertions
#endif
    //@fixme: test if it's faster if we always use the else branch
    if(activemask == ~0u) {
        assert(__shfl_sync(activemask, reinterpret_cast<const uintptr_t&>(acc), 0, warpSize) == reinterpret_cast<const uintptr_t&>(acc));
		assert(__activemask() == activemask);
        ValType sum = val;
#pragma unroll
        for (unsigned i = warp_size / 2; i != 0; i /= 2) {
            const ValType sum_other = __shfl_xor_sync(0xFFFFFFFFu, sum, i, warp_size);
            sum += sum_other;
        }
		assert(__shfl_sync(activemask, sum, 0, warpSize) == sum);
        if (get_lane_id() == 0) {
#ifndef NDEBUG
            if(!(std::abs(sum - sumRef) < 1E-1f || std::abs(sum - sumRef) / (std::max(std::abs(sum), std::abs(sumRef))) < 1E-1f)){
                printf("warpAtomicAdd: %f - %f\n", sum, sumRef);
            }
#endif
            atomicAdd(acc, sum);
        }
    }
    else{
        ValType sum = val;
        const auto g = cg::coalesced_threads();
        assert(g.shfl(reinterpret_cast<const uintptr_t&>(acc), 0) == reinterpret_cast<const uintptr_t&>(acc));
        const unsigned thrds = g.size();
        if (thrds != 1){
            const unsigned xorMask = ((0x1u << 31) >> __clz(thrds - 1));
            assert(thrds > xorMask && thrds <= xorMask * 2);
            {
                ValType sum_other = g.shfl(sum, std::min(g.thread_rank() ^ xorMask, thrds - 1));
                if ((g.thread_rank() ^ xorMask) < thrds) {
                    sum += sum_other;
                }
            }
            if (g.thread_rank() < xorMask) {
                const auto g1 = cg::coalesced_threads();
                assert(g1.size() == xorMask);
                for (int i = g1.size() / 2; i != 0; i /= 2) {
                    const ValType sum_other = g1.shfl(sum, g.thread_rank() ^ i);
                    sum += sum_other;
                }
            }
        }
        if (g.thread_rank() == 0) {
            assert(std::abs(sum - sumRef) < 1E-1f || std::abs(sum - sumRef) / (std::max(std::abs(sum), std::abs(sumRef))) < 1E-1f);
            atomicAdd(acc, sum);
        }
    }
#endif
}
#endif

#if 0
// bisect in [lower, upper)
template <typename ValType, typename IdxType = uint32_t>
__host__ __device__ inline
IdxType bisect(const ValType* __restrict__ basePtr, IdxType lower, IdxType upper, const ValType val){
    assert(lower < upper && basePtr[lower] <= val && basePtr[upper - 1] >= val);
    while (lower + 1 < upper){
        const IdxType mid = (lower + upper) / 2;
        if (basePtr[mid] <= val){
            lower = mid;
        }
        else{
            upper = mid;
        }
    }
    return lower;
}

// find in [lower, upper)
template <typename ValType, typename IdxType = uint32_t>
__host__ __device__ inline
IdxType findVal(const ValType* __restrict__ basePtr, IdxType lower, IdxType upper, const ValType val, IdxType initStep = 1){
    assert(basePtr[lower] <= val && basePtr[upper - 1] >= val);
    {
        // find a new upper bound first
        IdxType step = std::max(1, initStep);
        IdxType pos = lower;
        while (basePtr[pos] < val) {
            lower = pos + 1;
            pos += step;
            if (pos >= upper) {
                pos = upper - 1;
                break;
            }
            step *= 2;
        }
        upper = pos + 1;
    }
    // bisect
    return bisect(basePtr, lower, upper, val);
}
#endif


template <typename DataType, typename Fetcher>
class Prefetcher
{
public:
    __device__ inline
    explicit Prefetcher(Fetcher&& fetcher) : _fetcher{std::move(fetcher)} {}
    __device__ inline
    DataType get() {return _data[0];}
    template <typename ... Args>
    __device__ inline
    void fetch(Args... args){
        static_assert(std::is_same<DataType, typename std::result_of<Fetcher(Args...)>::type>::value, "fatal error");
        _data[0] = _data[1];
        _data[1] = _fetcher(args...);
    }
private:
    Fetcher _fetcher;
    DataType _data[2];
};

template <typename Func, typename... Args>
cudaError_t launchKernel(Func func, dim3 gridShape, dim3 ctaShape, size_t smemBytes, cudaStream_t stream, Args&&... args);

#if defined(__CUDACC__)
template <typename Func, typename... Args>
cudaError_t launchKernel(Func func, dim3 gridShape, dim3 ctaShape, size_t smemBytes, cudaStream_t stream, Args&&... args)
{
    if (gridShape.x == 0 || gridShape.y == 0 || gridShape.z == 0) {
        return cudaPeekAtLastError();
    }
    func<<<gridShape, ctaShape, smemBytes, stream>>>(std::forward<Args&&>(args)...);
    return cudaGetLastError();
}
#endif
