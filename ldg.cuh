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
// Created by yao on 9/7/19.
//

#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

template <size_t size>
struct LdgType;

template <>
struct LdgType<1>
{
    using Type = uint8_t;
};
template <>
struct LdgType<2>
{
    using Type = uint16_t;
};
template <>
struct LdgType<4>
{
    using Type = uint32_t;
};
template <>
struct LdgType<8>
{
    using Type = float2;
};
template <>
struct LdgType<16>
{
    using Type = float4;
};

// Note that ldg() may cause usage of stack memory. Use it with care.
template <typename T>
__device__ __forceinline__
T ldg(const T* __restrict__ ptr)
{
#if 1
    static_assert(sizeof(T) == alignof(T));
    constexpr uint32_t loadSize = unsigned(sizeof(T));
    typename LdgType<loadSize>::Type buffer;
    static_assert(sizeof(T) == sizeof(buffer), "fatal error");
    buffer = __ldg(reinterpret_cast<const typename LdgType<loadSize>::Type*>(ptr));
    T result;
    result = reinterpret_copy<T>(buffer);
    return result;
#else
    return *ptr;
#endif
}
