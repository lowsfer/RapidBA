//
// Created by yao on 20/10/18.
//

#pragma once
#include "platform.h"
#include <type_traits>
#include <cuda_runtime_api.h>
#include <stdexcept>
template <typename DataType, bool isConst = false>
using ConstSelector = typename std::conditional<isConst, const DataType , DataType>;

template <typename DataType, bool isConst = false>
using ConstSelectType = typename ConstSelector<DataType, isConst>::type;

template <typename T>
__host__ __device__ inline
constexpr T divUp(T a, T b){
    return (a+b-1)/b;
}

template <typename T>
__host__ __device__ inline
constexpr T roundUp(T a, T b){
    return divUp(a, b) * b;
}

template <typename DstType, typename SrcType>
__host__ __device__ inline
DstType reinterpret_copy(const SrcType& src){
    static_assert (sizeof(DstType) == sizeof(SrcType) && std::is_trivially_copyable<DstType>::value && std::is_trivially_copyable<SrcType>::value, "Invalid reinterpret_copy");
    DstType dst;
    memcpy(&dst, &src, sizeof(DstType));
//    dst = reinterpret_cast<const DstType&>(src);
    return dst;
}

template <typename T>
__host__ __device__ inline
void unused(const T& arg){
    static_cast<void>(arg);
}

template <typename T, typename... Args>
__host__ __device__ inline
void unused(const T& arg0, const Args&... args)
{
    unused(arg0);
    unused(args...);
}

template <typename T, size_t size>
constexpr bool arrayEqual(const T(&a)[size], const T(&b)[size])
{
    for (size_t i = 0; i < size; i++){
        if (a[i] != b[i]){
            return false;
        }
    }
    return true;
}

namespace sym {
struct Zero {
    template <typename T> __device__ __host__ __forceinline__
    constexpr explicit operator T() const {return T{0};}
};

template<typename T>
__device__ __host__ __forceinline__
constexpr Zero operator*(const Zero &a, const T &b) { return Zero{}; }

template<typename T, std::enable_if_t<!std::is_same<T,Zero>::value, int> = 0>
__device__ __host__ __forceinline__
constexpr Zero operator*(const T &a, const Zero &b) { return Zero{}; }

template<typename T>
__device__ __host__ __forceinline__
constexpr T operator+(const Zero &a, const T &b) { return b; }

template<typename T, std::enable_if_t<!std::is_same<T,Zero>::value, int> = 0>
__device__ __host__ __forceinline__
constexpr T operator+(const T &a, const Zero &b) { return a; }

struct One {
    template <typename T> __device__ __host__ __forceinline__
    constexpr explicit operator T() const {return T{1};}
};
template<typename T>
__device__ __host__ __forceinline__
constexpr T operator*(const One &a, const T &b) { return b; }

template<typename T, std::enable_if_t<!std::is_same<T,One>::value, int> = 0>
__device__ __host__ __forceinline__
constexpr T operator*(const T &a, const One &b) { return a; }

template<typename T>
__device__ __host__ __forceinline__
constexpr T operator+(const One &a, const T &b) { return T(a) + b; }

template<typename T, std::enable_if_t<!std::is_same<T,One>::value, int> = 0>
__device__ __host__ __forceinline__
constexpr T operator+(const T &a, const One &b) { return a + T(b); }

} // namespace sym

#ifdef __CUDA_ARCH__
__device__ __forceinline__
void fail() {
    assert(false);
    asm volatile("trap;\n");
}
#else
[[noreturn]] inline void fail() {
    assert(false);
    throw std::logic_error("This shall never happen");
}
#endif

template<class Lambda, int=(Lambda{}(), 0)>
constexpr bool is_constexpr(Lambda) { return true; }
constexpr bool is_constexpr(...) { return false; }

template <typename Index>
constexpr Index badIdx(){
    static_assert(std::is_integral<Index>::value, "Index must be a integer type");
    if (std::is_unsigned<Index>::value){
        return std::numeric_limits<Index>::max();
    }
    else if(std::is_signed<Index>::value){
        return static_cast<Index>(-1);
    }
    else{
        fail();
    }
    return static_cast<Index>(-1);
}

template <typename T>
constexpr bool alwaysFalse() {return false;}
