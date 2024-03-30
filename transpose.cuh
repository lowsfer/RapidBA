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
// Created by yao on 12/11/18.
//

#pragma once
#include "cuda_hint.cuh"
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cooperative_groups.h>
#include <type_traits>
#include <cassert>

namespace cg = cooperative_groups;

template <int nbRegs = 32, int grpSize = 32>
struct GroupLoadTransposer{
    static_assert(nbRegs <= grpSize, "fatal error");
    using Reg = uint32_t;
    using Buffer = Reg[nbRegs][grpSize];

    __device__ __forceinline__
    GroupLoadTransposer(Buffer& smem) : buffer{smem}{}

    __device__ __forceinline__
    void store_sync(const cg::thread_block_tile<grpSize>& g, const Reg (&src)[grpSize]){
        assert(grpSize != warpSize || __activemask() == ~0u);
        if (g.thread_rank() < nbRegs){
            for (int i = 0; i < grpSize; i++) {
                buffer[g.thread_rank()][(g.thread_rank() + i) % grpSize] = src[i];
            }
        }
        g.sync();
    }
    __device__ __forceinline__
    void load_sync(const cg::thread_block_tile<grpSize>& g, Reg (&dst)[nbRegs]){
        assert(grpSize != warpSize || __activemask() == ~0u);
        for (int i = 0; i < nbRegs; i++) {
            dst[i] = buffer[i][(g.thread_rank() + i) % grpSize];
        }
        g.sync();
    }
    __device__ __forceinline__
    void transpose(const cg::thread_block_tile<grpSize>& g, Reg (&dst)[nbRegs], const Reg (&src)[grpSize]){
        store_sync(g, src);
        load_sync(g, dst);
    }

    Buffer& buffer;
};

template <typename DstType, typename SrcType>
__device__ __forceinline__
DstType raw_cast(const SrcType& src){
    static_assert(sizeof(DstType) == sizeof(SrcType), "invalid cast");
    DstType dst;
    memcpy(&dst, &src, sizeof(SrcType));
    return dst;
}

template <int nbRegs = 32, int grpSize = 32, typename RegType = float>
struct GroupStoreTransposer{
    static_assert(nbRegs <= grpSize, "fatal error");
    using Reg = RegType;
    using Buffer = Reg[nbRegs][grpSize];

    __device__ __forceinline__
    GroupStoreTransposer(Buffer& smem) : buffer{smem}{}

    __device__ __forceinline__
    void store_sync(const cg::thread_block_tile<grpSize>& g, const Reg (&src)[nbRegs]){
        for (int i = 0; i < nbRegs; i++) {
            buffer[i][(g.thread_rank() + i) % grpSize] = src[i];
        }
        g.sync();
        assert(grpSize != warpSize || __activemask() == ~0u);
    }
    __device__ __forceinline__
    Reg load_one(const cg::thread_block_tile<grpSize>& g, int i){
        assert(g.thread_rank() < nbRegs);
        return buffer[g.thread_rank()][(g.thread_rank() + i) % grpSize];
    }
    __device__ __forceinline__
    void load_sync(const cg::thread_block_tile<grpSize>& g, Reg (&dst)[grpSize]){
        assert(grpSize != warpSize || __activemask() == ~0u);
        if (g.thread_rank() < nbRegs){
            for (int i = 0; i < grpSize; i++) {
                dst[i] = load(g, i);
            }
        }
        g.sync();
    }

    __device__ __forceinline__
    void transpose(const cg::thread_block_tile<grpSize>& g, Reg (&dst)[grpSize], const Reg (&src)[nbRegs]){
        store_sync(g, src);
        load_sync(g, dst);
    }

    Buffer& buffer;
};



template <int nbRegs, int grpSize>
struct GroupReducerImpl {
    using Buffer = typename GroupStoreTransposer<nbRegs, grpSize>::Buffer;
    __device__ inline static
    void reduce(const cg::thread_block_tile<grpSize> &g, Buffer& buff, float* dst, const float* src /*[nbRegs]*/) {
        GroupStoreTransposer<nbRegs, grpSize, float> trans(buff);
        trans.store_sync(g, *reinterpret_cast<const float(*)[nbRegs]>(src));
        float acc = 0;
        if (nbRegs == grpSize || g.thread_rank() < nbRegs){
            for (int i = 0; i < grpSize; i++) {
                acc += trans.load_one(g, i);
            }
        }
#ifndef NDEBUG
        else{
            acc = NAN;
        }
#endif
        g.sync();
        dst[0] = acc;
    }
};

template <int grpSize>
struct GroupReducerImpl<0, grpSize>{
    struct Buffer{};
    __device__ inline static
    void reduce(const cg::thread_block_tile<grpSize> &g, Buffer& buff, float* dst, const float* src /*[nbRegs]*/){
        dst[0] = 0;
    }
};

template <int nbRegs, int grpSize>
struct GroupReducerLarge {
    static_assert(nbRegs > grpSize, "Should use GroupReducerImpl instead");
    union Buffer{
        typename GroupStoreTransposer<grpSize, grpSize>::Buffer full;
        typename GroupStoreTransposer<nbRegs%grpSize, grpSize>::Buffer res;
    };

    __device__ inline static
    void reduce(const cg::thread_block_tile<grpSize> &g, Buffer& buff, float* dst, const float* src /*[nbRegs]*/) {
        for (int i = 0; i < nbRegs / grpSize; i++){
            GroupReducerImpl<grpSize, grpSize>::reduce(g, buff.full, &dst[i], &src[grpSize*i]);
        }
        if (nbRegs % grpSize != 0){
            GroupReducerImpl<nbRegs%grpSize, grpSize>::reduce(g, buff.res, &dst[nbRegs/grpSize], &src[grpSize*(nbRegs/grpSize)]);
        }
    }
};

template <int nbRegs, int grpSize>
using GroupReducer = typename std::conditional<std::less<int>{}(nbRegs, grpSize), GroupReducerImpl<nbRegs, grpSize>, GroupReducerLarge<nbRegs, grpSize>>::type;
