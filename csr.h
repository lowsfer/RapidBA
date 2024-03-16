//
// Created by yao on 17/10/18.
//

#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>
#include <cassert>
#include "utils_general.h"

namespace rba {
template<typename DataType, typename SizeType = uint32_t>
struct CSR {
    uint32_t nbRows;
    SizeType nbNonZeroElems; // equal to rows[nbRows]
    const SizeType *__restrict__ rows; // length = nbRows + 1, rows[0] = 0
    const uint32_t *__restrict__ idxCol; //col index, length = rows[nbRows]
    DataType *__restrict__ data; // length = rows[nbRows]

    __device__ __host__ inline
    const SizeType& nbElems() const {
#if __CUDACC__
        assert(nbNonZeroElems == rows[nbRows]);
#endif
        return nbNonZeroElems;
    }
    __device__ __host__ inline
    SizeType getRowBegin(uint32_t idxRow) const {
        assert(idxRow < nbRows);
        return __ldg(&rows[idxRow]);
    }
    __device__ __host__ inline
    SizeType getRowEnd(uint32_t idxRow) const {
        assert(idxRow < nbRows);
        return __ldg(&rows[idxRow+1]);
    }
    __device__ __host__ inline
    SizeType getRowLength(uint32_t idxRow) const {return getRowEnd(idxRow) - getRowBegin(idxRow);}
    __device__ __host__ inline
    const DataType* operator[](uint32_t idxRow) const {return &data[getRowBegin(idxRow)];}
    __device__ __host__ inline
    DataType* operator[](uint32_t idxRow) {return &data[getRowBegin(idxRow)];}
};

// std::vector<std::vector<DataType>> on GPU
template <typename DataType, uint32_t alignment = 128, typename SizeType = uint32_t>
class VecVec  {
public:
    static_assert(alignment % sizeof(DataType) == 0, "cannot align");

    __device__ __host__ inline
    VecVec():VecVec(0, nullptr, nullptr){}
    __device__ __host__ inline
    VecVec(uint32_t nbRows, const SizeType* __restrict__ rows, DataType* __restrict__ data)
    :nbRows{nbRows}, rows{rows}, data{data}{}

    __device__ __host__ inline
    uint32_t getNbRows() const {return nbRows;}
    // access the row
    __device__ __host__ inline
    const DataType* operator[](uint32_t idxRow) const {return &data[getRowBegin(idxRow)];}
    __device__ __host__ inline
    DataType* operator[](uint32_t idxRow) {return &data[getRowEnd(idxRow)];}

    __device__ __host__ inline
    SizeType getRowSize(uint32_t idxRow) const {return getRowEnd(idxRow) - getRowBegin(idxRow);}

private:
    __device__ __host__ inline
    SizeType getRowBegin(uint32_t idxRow) const {
        assert(idxRow < nbRows);
        assert(reinterpret_cast<const uintptr_t&>(const_cast<typename std::remove_cv<DataType*&>::type>(data)) % alignment == 0);
        return roundUp(ldg(&rows[idxRow]), std::uint32_t(alignment/sizeof(DataType)));
    }
    __device__ __host__ inline
    SizeType getRowEnd(uint32_t idxRow) const {
        assert(idxRow < nbRows);
        return ldg(&rows[idxRow+1]);
    }
private:
    const uint32_t nbRows;
    // length = nbRows + 1, rows[0] = 0. Each row starts from roundUp(rows[i], alignment/sizeof(DataType)) and ends at rows[i+1]
    // We make it private to prevent direct accessing
    const SizeType* __restrict__ rows;
    DataType *__restrict__ data;
};


}