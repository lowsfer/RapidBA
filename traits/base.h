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
// Created by yao on 9/3/19.
//

#pragma once
#include <cstdint>

namespace rba {

using IdxPt = uint32_t;
using IdxCap = uint32_t;
using IdxCam = uint32_t;
using IdxOb = uint32_t;
using IdxMBlock = IdxCam;
using IdxUBlock = IdxCap;
using IdxVBlock = IdxPt;

namespace impl{
template <typename Dst, typename Src>
struct Caster
{
    static constexpr Dst cast(Src src) {return static_cast<Dst>(src);}
};
template <typename T>
struct Caster<T, T>
{
    static constexpr T cast(T src) {return src;}
};
}

template <typename ftype>
struct Coordinate
{
    using ImplType = float;
    ImplType value; //@info: double here causes 15% perf drop compared with float
    __host__ __device__ __forceinline__ ftype operator-(const Coordinate<ftype>& rhs) const {return static_cast<ftype>(value - rhs.value);}
    __host__ __device__ __forceinline__ Coordinate<ftype> operator+(ftype delta) const {return {value + delta};}
    __host__ __device__ __forceinline__ Coordinate<ftype>& operator+=(ftype delta) { value += delta; return *this; }
    __host__ __device__ __forceinline__ Coordinate<ftype> operator-(ftype delta) const {return {value - delta};}
    __host__ __device__ __forceinline__ Coordinate<ftype>& operator-=(ftype delta) { value -= delta; return *this; }
    __host__ __device__ __forceinline__ bool operator==(const Coordinate<ftype>& rhs) const { return value == rhs.value; }
    __host__ __device__ __forceinline__ bool operator!=(const Coordinate<ftype>& rhs) const { return value != rhs.value; }
    template <typename T>
    __host__ __device__ __forceinline__ T cast() const { return impl::Caster<T, ImplType>::cast(value); }
};

struct FPTraitsSSD {
    using lpf = float;
    using hpf = float;
    using epf = double;
    using locf = Coordinate<lpf>; // floating-point type for location, i.e. point/camera coordinates
    using ObIdx = IdxOb;
};

} // namespace rba

#define RBA_IMPORT_FPTYPES(FPTraits) \
    using lpf = typename FPTraits::lpf; \
    using hpf = typename FPTraits::hpf; \
    using epf = typename FPTraits::epf; \
    using locf = typename FPTraits::locf
