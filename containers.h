//
// Created by yao on 28/11/18.
//

#pragma once
#include "csr.h"
#include "kmat.h"
#include <cstdint>
#include "utils_host.h"

// CSR format
template <typename DataType>
struct DevCSR{
    DevCSR():nbRows{0u}, rows{0u}{assert(rows.size() == 1 && rows.at(0) == 0);}
    uint32_t nbRows;
    MngVector<uint32_t> rows;
    MngVector<uint32_t> idxCol;
    MngVector<DataType> data;

    size_t nbNonZeroElems() const {
        // do not use data.size() as for W block, we don't need the data and did not allocate memory.
        return idxCol.size();
    }

    void clear() {
        nbRows = 0;
        rows = MngVector<uint32_t>({0U});
        assert(rows.at(0) == 0);
        idxCol.clear();
        data.clear();
    }

    rba::CSR<DataType> getMutable() {
        assert(nbRows + 1 == rows.size());
        return {nbRows, cast32u(nbNonZeroElems()), rows.data(), idxCol.data(), data.empty() ? nullptr : data.data()};
    }
    rba::CSR<const DataType> getConst() const {
        assert(nbRows + 1 == rows.size());
        return {nbRows, cast32u(nbNonZeroElems()), rows.data(), idxCol.data(), data.empty() ? nullptr : data.data()};
    }

    template <typename Derived, bool enabler = true, std::enable_if_t<enabler && is_kmat<DataType>::value, int> = 1>
    void assignData(const Eigen::MatrixBase<Derived>& src){
        assert(src.rows() == nbRows * DataType::rows());
        assert(src.cols() % DataType::cols() == 0);
        for (uint32_t i = 0; i < nbRows; i++){
            DataType* row = &data[rows[i]];
            const uint32_t* cols = &idxCol[rows[i]];
            const auto rowSize = rows[i+1] - rows[i];
            auto srcBlock = [&](uint32_t col){return src.template block<DataType::rows(), DataType::cols()>(DataType::rows() * i, DataType::cols() * col);};
            auto checkZeros = [&](uint32_t colBeg, uint32_t colEnd){
                for (uint32_t col = colBeg; col < colEnd; col++){
                    assert(srcBlock(col).squaredNorm() < 1E-2f);
                }
            };
            checkZeros(0, cols[0]);
            for (uint32_t j = 0; j < rowSize; j++){
                toEigenMap(row[j]) = srcBlock(cols[j]).template cast<typename DataType::ValType>();
                if (j + 1 < rowSize)
                    checkZeros(cols[j] + 1, cols[j+1]);
            }
            checkZeros(cols[rowSize - 1] + 1, src.cols() / DataType::cols());
        }
    }

    cudaError_t migrateToDevice(int deviceId, cudaStream_t stream) const{
        checkEarlyReturn(rows.migrateToDevice(deviceId, stream));
        checkEarlyReturn(idxCol.migrateToDevice(deviceId, stream));
        checkEarlyReturn(data.migrateToDevice(deviceId, stream));
        return cudaSuccess;
    }

private:
    template <typename EType>
    struct Elem{
        using type = EType;
    };
    template <typename EType, std::uint32_t Rows, std::uint32_t Cols>
    struct Elem<kmat<EType, Rows, Cols>>{
        using type = EType;
    };
public:

    template <typename ValType = typename Elem<DataType>::type>
    std::enable_if_t<is_kmat<DataType>::value, Eigen::Matrix<ValType, -1, -1>> toEigen(size_t colsInBlocks) const {
        const uint32_t blockRows = DataType::rows();
        const uint32_t blockCols = DataType::cols();
        Eigen::Matrix<ValType, -1, -1> dst = Eigen::Matrix<ValType, -1, -1>::Zero(blockRows * nbRows, blockCols * colsInBlocks);
        for (uint32_t i = 0; i < nbRows; i++){
            const DataType* row = &data[rows[i]];
            const uint32_t* cols = &idxCol[rows[i]];
            const auto rowSize = rows[i+1] - rows[i];
            auto dstBlock = [&](uint32_t col){return dst.template block<blockRows, blockCols>(blockRows * i, blockCols * col);};
            for (uint32_t j = 0; j < rowSize; j++){
                dstBlock(cols[j]) = toEigenMap(row[j]).template cast<ValType>();
            }
        }
        return dst;
    }
};

template <typename DataType, uint32_t alignment = 128, typename SizeType = uint32_t>
class DevVecVec{
public:
    DevVecVec() : _nbRows{0}, _rows{{0}} {assert(_rows.size() == 1 && _rows.at(0) == 0);}
    void appendRow(const DataType* row, SizeType rowLength, DataType padVal = std::numeric_limits<DataType>::max()){
        const uint32_t rowBeg = roundUp(_rows[_nbRows], nbAlignedItems);
        assert(_data.size() == rowBeg);
        _rows.push_back(rowBeg + rowLength);
        _data.insert(_data.end(), row, row+rowLength);
        uint32_t pad = roundUp(rowLength, nbAlignedItems) - rowLength;
        for (uint32_t i = 0; i < pad; i++)
            _data.emplace_back(padVal);
        assert(_data.size() == roundUp(_rows.back(), nbAlignedItems));
        _nbRows++;
    }
    SizeType rows()const{return _nbRows;}
    const DataType* row(uint32_t idxRow) const {return &_data[roundUp(_rows[idxRow], nbAlignedItems)];}
    uint32_t rowSize(uint32_t idxRow) const {return _rows[idxRow + 1] - roundUp(_rows[idxRow], nbAlignedItems);}
    const DataType* data()const{return _data.data();}
//    rba::VecVec<DataType, alignment, SizeType> getMutable() {
//        return {_nbRows, _rows.data(), _data.data()};
//    }
    rba::VecVec<const DataType, alignment, SizeType> getConst() const {
        return {_nbRows, _rows.data(), _data.data()};
    }
    void reserveRows(size_t nbRows) {_rows.reserve(nbRows + 1);}
    // User should consider padding after each row
    void reserveData(size_t nbElems) {_data.reserve(nbElems + 1);}
    void clear() {_nbRows = 0; _rows.clear(); _rows.resize(_nbRows+1); assert(_rows.at(0) == 0); _data.clear();}
    constexpr static uint32_t nbAlignedItems = alignment / sizeof(DataType);

    cudaError_t migrateToDevice(int deviceId, cudaStream_t stream) const {
        checkEarlyReturn(_rows.migrateToDevice(deviceId, stream));
        checkEarlyReturn(_data.migrateToDevice(deviceId, stream));
        return cudaSuccess;
    }

    DevVecVec getHostCopy() const {
        DevVecVec result;
        result._nbRows = _nbRows;
        result._rows = _rows.getHostCopy();
        result._data = _data.getHostCopy();
        return result;
    }
private:
    static_assert(alignment % sizeof(DataType) == 0, "fatal error");// for unaligned data, just use alignment = sizeof(DataType)
    uint32_t _nbRows;
    MngVector<SizeType> _rows;
    MngVector<DataType> _data;
};
