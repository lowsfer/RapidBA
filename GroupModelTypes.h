//
// Created by yao on 28/11/18.
//

#pragma once
#include "kernel.h"
#include "utils_host.h"
#include "containers.h"
#include "computeHessian.h"
#include "computeSchur.h"
#include "blockSolvers/solveSchurPCG.h"
#include "blockSolvers/solveHessian.h"
#include <memory>
#include "fwd.h"

namespace rba{
namespace grouped{

template <typename Traits>
struct DeviceModel {
    RBA_IMPORT_TRAITS(Traits);
    //varIdx == idxFixed means this point is fixed
    static constexpr uint32_t varIdxFixed = rba::ModelBase<Traits>::varIdxFixed;
    MngVector<typename rba::ModelBase<Traits>::CamIntrWVarIdx> intrinsics;
    MngVector<uint32_t> varIntri;

    MngVector<Capture> captures;
    MngVector<CtrlLoc<lpf>> gnssCtrl; // 1:1 mapped to captures[] if not nullptr. Typically dense or nullptr.
    MngVector<uint32_t> involvedCaps; // variable captures first, then involved fixed captures.
    uint32_t nbVarCaps;

    MngVector<typename rba::ModelBase<Traits>::PointWVarIdx> points;
    MngVector<uint32_t> varPoints;

    struct {
        // for points[varPoints[idxVarPt[i]]], different from HostModel
        MngVector<uint32_t> idxVarPt;
        // Hard control points should use fixed points instead.
        MngVector<CtrlLoc<lpf>> data;
    } softCtrlPoints;

    DevVecVec<typename rba::CapOb<lpf>, 128, IdxOb> capObs;

    static void syncStream(cudaStream_t stream){
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
    void clear() {
        intrinsics.clear();
        varIntri.clear();
        captures.clear();
        involvedCaps.clear();
        nbVarCaps = 0;
        points.clear();
        varPoints.clear();
        capObs.clear();
    }
    void init(const HostModel<Traits>& src);
    void saveToHost(HostModel<Traits>& dst) const;
    cudaError_t migrateToDevice(int deviceId, cudaStream_t stream) const;

    template <bool isConst = true>
    rba::Model<Traits, isConst> getKernelArgs() {
        assert(softCtrlPoints.idxVarPt.size() == softCtrlPoints.data.size());
        return rba::Model<Traits, isConst>{
                uint32_t(intrinsics.size()), uint32_t(captures.size()), uint32_t(points.size()),
                intrinsics.data(),
                {varIntri.data(), uint32_t(varIntri.size())},
                captures.data(),
                gnssCtrl.empty() ? nullptr : gnssCtrl.data(),
                {nbVarCaps, uint32_t(involvedCaps.size()), involvedCaps.data()},
                points.data(),
                {varPoints.data(), uint32_t(varPoints.size())},
                {softCtrlPoints.idxVarPt.data(), softCtrlPoints.data.data(), cast32u(softCtrlPoints.idxVarPt.size())},
                capObs.getConst()
        };
    }

    rba::Model<Traits, true> getKernelArgs() const {
        assert(softCtrlPoints.idxVarPt.size() == softCtrlPoints.data.size());
        return rba::Model<Traits, true>{
                uint32_t(intrinsics.size()), uint32_t(captures.size()), uint32_t(points.size()),
                intrinsics.data(),
                {varIntri.data(), uint32_t(varIntri.size())},
                captures.data(),
                gnssCtrl.empty() ? nullptr : gnssCtrl.data(),
                {nbVarCaps, uint32_t(involvedCaps.size()), involvedCaps.data()},
                points.data(),
                {varPoints.data(), uint32_t(varPoints.size())},
                {softCtrlPoints.idxVarPt.data(), softCtrlPoints.data.data(), cast32u(softCtrlPoints.idxVarPt.size())},
                capObs.getConst()
        };
    }

    using VarBackup = DeviceModelVarBackup<Traits>;
    void update(VarBackup& backup, const DevHessianVec<Traits>& delta, cudaStream_t stream);
    void revert(const VarBackup& backup, cudaStream_t stream);
};

template <typename Traits>
struct DeviceModelVarBackup
{
    RBA_IMPORT_TRAITS(Traits);
    DevVector<CamIntr> intrinsics;
    DevVector<Capture> captures;
    DevVector<rba::Point<lpf>> points;

    void clear() {
        resize(0, 0, 0);
    }

    void resize(const DeviceModel<Traits>& model){
        resize(model.varIntri.size(), model.nbVarCaps, model.varPoints.size());
    }
    void resize(size_t nbVarIntri, size_t nbVarCap, size_t nbVarPoints){
        intrinsics.resize(nbVarIntri);
        captures.resize(nbVarCap);
        points.resize(nbVarPoints);
    }

    template <bool isConst>
    using BackupData = rba::updateModel::BackupData<Traits, isConst>;
    template <bool isConst = true>
    BackupData<isConst> getParams() {
        return {intrinsics.data(), cast32u(intrinsics.size()), captures.data(), cast32u(captures.size()), points.data(), cast32u(points.size())};
    }
    BackupData<true> getParams() const {
        return {intrinsics.data(), cast32u(intrinsics.size()), captures.data(), cast32u(captures.size()), points.data(), cast32u(points.size())};
    }
};


/** structure of Hessian equation
 * | M,  Q,  S |   | deltaC |   | Ec |
 * | Qt, U,  W | x | deltaA | = | Ea |
 * | St, Wt, V |   | deltaB |   | Eb |
 */

struct DeviceHessianPreCompData{
    uint32_t nbVarCaps;
    uint32_t nbInvolvedCaps;
    // shape = (nbVarCaps or nbInvolvedCaps, nbObsInCap). These are LUTs, not really CSR
    using IdxObVarCap = rba::IdxOb;
    MngVector<rba::IdxOb> rows;// length = nbInvolvedCaps+1
    MngVector<uint32_t> sparseIdxS; //length = rows[nbInvolvedCaps]
    MngVector<uint32_t> sparseIdxW; //length = rows[nbVarCaps]

    void clear() {
        nbVarCaps = 0;
        nbInvolvedCaps = 0;
        rows.clear();
        sparseIdxS.clear();
        sparseIdxW.clear();
    }

    cudaError_t migrateToDevice(int deviceId, cudaStream_t stream) const;

    rba::HessianPreCompData<true> getKernelArgs() const {
        return {
                nbVarCaps,
                nbInvolvedCaps,
                rows.data(),
                sparseIdxS.data(),
                sparseIdxW.data()
        };
    }
};

template <typename Traits>
struct DeviceHessian{
#define USING_HESSIAN_BLOCK_TYPE(XBlock) using XBlock = typename HessianBase<Traits>::XBlock
    USING_HESSIAN_BLOCK_TYPE(MBlock);
    USING_HESSIAN_BLOCK_TYPE(MSymBlock);
    USING_HESSIAN_BLOCK_TYPE(EcBlock);
    USING_HESSIAN_BLOCK_TYPE(UBlock);
    USING_HESSIAN_BLOCK_TYPE(USymBlock);
    USING_HESSIAN_BLOCK_TYPE(EaBlock);
    USING_HESSIAN_BLOCK_TYPE(VBlock);
    USING_HESSIAN_BLOCK_TYPE(VSymBlock);
    USING_HESSIAN_BLOCK_TYPE(EbBlock);
    USING_HESSIAN_BLOCK_TYPE(QBlock);
    USING_HESSIAN_BLOCK_TYPE(SBlock);
    USING_HESSIAN_BLOCK_TYPE(WBlock);
#undef USING_HESSIAN_BLOCK_TYPE

    MngVector<MSymBlock> M;
    MngVector<EcBlock> Ec;

    MngVector<USymBlock> U;
    MngVector<EaBlock> Ea;

    MngVector<VSymBlock> V;
    MngVector<EbBlock> Eb;

    //see Hessian<false>::Q. we don't use CSR for Q. We use one dense row instead.
    struct {
        MngVector<QBlock> blocks;
        MngVector<IdxMBlock> row;
    } Q;

    DevCSR<SBlock> S;

    DevCSR<WBlock> W;

    void clear() {
        M.clear();
        Ec.clear();
        U.clear();
        Ea.clear();
        V.clear();
        Eb.clear();
        Q.blocks.clear();
        Q.row.clear();
        S.clear();
        W.clear();
    }

    void init(const HostModel<Traits> &model, const rba::IdxCap *involvedCaps, size_t nbInvolvedCaps,
                  DeviceHessianPreCompData *preComp, bool computeHessianW = false);
    rba::Hessian<Traits, false> getKernelArgs() {
        return {
                uint32_t(M.size()),
                M.data(),
                Ec.data(),
                uint32_t(U.size()),
                U.data(),
                Ea.data(),
                uint32_t(V.size()),
                V.data(),
                Eb.data(),
                {Q.blocks.data(), Q.row.data()},
                S.getMutable(),
                W.getMutable()
        };
    }
    rba::Hessian<Traits, true> getKernelArgsConst() const {
        return {
                uint32_t(M.size()),
                M.data(),
                Ec.data(),
                uint32_t(U.size()),
                U.data(),
                Ea.data(),
                uint32_t(V.size()),
                V.data(),
                Eb.data(),
                {Q.blocks.data(), Q.row.data()},
                S.getConst(),
                W.getConst()
        };
    }

    rba::HessianVec<Traits, true> getGConst() const {
        return rba::HessianVec<Traits, true>{
            Ec.data(), cast32u(Ec.size()),
            Ea.data(), cast32u(Ea.size()),
            Eb.data(), cast32u(Eb.size())
        };
    }

    cudaError_t migrateToDevice(int deviceId, cudaStream_t stream) const;
};

struct DeviceSchurPreCompData;
template <typename Traits>
struct DeviceSchur{
#define USING_SCHUR_TYPE(type) using type = typename rba::Schur<Traits, false>::type
    USING_SCHUR_TYPE(MBlock);
    USING_SCHUR_TYPE(MSymBlock);
    USING_SCHUR_TYPE(EcBlock);
    USING_SCHUR_TYPE(QBlock);
    USING_SCHUR_TYPE(UBlock);
    USING_SCHUR_TYPE(USymBlock);
    USING_SCHUR_TYPE(EaBlock);
#undef USING_SCHUR_TYPE
    DevCSR<MBlock> upperM; //nbRows = nbVarIntri. Diagonal is not stored here. Last row is always empty
    MngVector<MSymBlock> diagM;
    MngVector<EcBlock> Ec;

    DevCSR<UBlock> upperU; //nbRows = nbVarCaps. Diagonal is not stored here.  Last row is always empty
    MngVector<USymBlock> diagU;
    MngVector<EaBlock> Ea;

    DevCSR<QBlock> Q;

    void clear() {
        upperM.clear();
        diagM.clear();
        Ec.clear();
        upperU.clear();
        diagU.clear();
        Ea.clear();
        Q.clear();
    }
    // pSparseIdxW2idxLocalOb is for optional optimization. If no provided, we will work out one.
    void init(const DeviceModel<Traits> &model, const DeviceHessian<Traits> &hessian, DeviceSchurPreCompData *precomp,
                  std::shared_ptr<DevVecVec<uint16_t>> pSparseIdxW2idxLocalOb = nullptr);
    rba::Schur<Traits, false> getKernelArgs(){
        return {
                rba::IdxMBlock(diagM.size()),
                upperM.getMutable(),
                diagM.data(),
                Ec.data(),
                rba::IdxUBlock(diagU.size()),
                upperU.getMutable(),
                diagU.data(),
                Ea.data(),
                Q.getMutable()
        };
    }
    rba::Schur<Traits, true> getKernelArgsConst() const{
        return {
                rba::IdxMBlock(diagM.size()),
                upperM.getConst(),
                diagM.data(),
                Ec.data(),
                rba::IdxUBlock(diagU.size()),
                upperU.getConst(),
                diagU.data(),
                Ea.data(),
                Q.getConst()
        };
    }
    cudaError_t migrateToDevice(int deviceId, cudaStream_t stream) const;
};

struct DeviceSchurPreCompData{
    MngVector<rba::IdxMBlock> rowTableUpperM;

    MngVector<rba::IdxUBlock> rowTableUpperU;

    // for upperM
    DevVecVec<rba::SchurUpperMComputer::PreComp::PairSS> pairSS;

    // for upperU
    DevVecVec<rba::SchurUpperUComputer::ObPair> pairWW;

    // for diagU
    std::shared_ptr<DevVecVec<uint16_t>> sparseIdxW2idxLocalOb; // list of local ob index for variable points. nbRows = nbUpperUItems

    // for Q
    DevVecVec<uint32_t> qPairSparseIdxS;
    DevVecVec<uint16_t> qPairIdxLocalOb;
    MngVector<rba::IdxMBlock> rowTableQ;

    void clear() {
        rowTableUpperM.clear();
        rowTableUpperU.clear();
        pairSS.clear();
        pairWW.clear();
        if (sparseIdxW2idxLocalOb != nullptr)
            sparseIdxW2idxLocalOb->clear();
        qPairSparseIdxS.clear();
        qPairIdxLocalOb.clear();
        rowTableQ.clear();
    }

    rba::SchurPreComp toParams() const{
        return rba::SchurPreComp{
                {rowTableUpperM.data(), pairSS.getConst()},
                {rowTableUpperU.data(), pairWW.getConst()},
                sparseIdxW2idxLocalOb->getConst(),
                {qPairSparseIdxS.getConst(), qPairIdxLocalOb.getConst(), rowTableQ.data()}
        };
    }
    cudaError_t migrateToDevice(int deviceId, cudaStream_t stream) const;
};
template <typename Traits>
std::vector<std::vector<uint16_t>> make_sparseIdxW2idxLocalOb(const DeviceModel<Traits>& model, const DeviceHessian<Traits> &hessian);
void toDevice_sparseIdxW2idxLocalOb(DevVecVec<uint16_t> &dst, const std::vector<std::vector<uint16_t>> &src);
std::vector<std::vector<uint16_t>> toHost_sparseIdxW2idxLocalOb(const DevVecVec<uint16_t>& src);


struct DevicePCGPreComp{
    //store index to Schur::upperM::data
    DevCSR<uint32_t> lowerM;//nbRows == Schur::nbVarIntri. First row is always empty
    DevCSR<uint32_t> lowerU;//nbRows == Schur::nbVarCap. First row is always empty
    DevCSR<uint32_t> lowerQ;

    void clear() {
        lowerM.clear();
        lowerU.clear();
        lowerQ.clear();
    }
    template <typename Traits>
    void init(const rba::grouped::DeviceSchur<Traits> &devSchur);
    cudaError_t migrateToDevice(int deviceId, cudaStream_t stream) const;
};

//@fixme: add unit test
template <typename DataType>
void transposeCSR(DevCSR<uint32_t>& dst, const DevCSR<DataType>& src, uint32_t nbCols){
    dst.clear();

    struct Pair{
        uint32_t idxCol;
        uint32_t idxData;
    };
    std::vector<std::vector<Pair>> cols(nbCols);
    for (uint32_t i = 0; i < src.nbRows; i++){
        const uint32_t rowBeg = src.rows[i];
        const uint32_t* idxCol = &src.idxCol[rowBeg];
        uint32_t rowSize = src.rows[i+1] - rowBeg;
        for (uint32_t j = 0; j < rowSize; j++){
            assert(idxCol[j] < nbCols);
            cols[idxCol[j]].emplace_back(Pair{i, rowBeg + j});
        }
    }

    dst.nbRows = nbCols;
    dst.rows.resize(nbCols + 1);
    dst.rows[0] = 0;
    for (uint32_t i = 0; i < nbCols; i++)
        dst.rows[i+1] = dst.rows[i] + cast32u(cols[i].size());
    dst.idxCol.clear();
    dst.data.clear();
    dst.idxCol.reserve(dst.rows.back());
    dst.data.reserve(dst.rows.back());
    for (uint32_t i = 0; i < nbCols; i++){
        assert(std::is_sorted(cols[i].begin(), cols[i].end(), [](const Pair& a, const Pair& b){return a.idxCol < b.idxCol;}));
        for (const Pair& p : cols[i]){
            dst.idxCol.push_back(p.idxCol);
            dst.data.push_back(p.idxData);
        }
        assert(dst.data.size() == dst.rows[i+1]);
    }
}

template <typename Traits>
struct DevSchurVec
{
    void clear() {c.clear(); a.clear();}
    void resize(uint32_t sizeC, uint32_t sizeA){
        clear();
        c.resize(sizeC);
        a.resize(sizeA);
    }
    pcg::SchurVec<Traits, true> getParamConst() const {
        return pcg::SchurVec<Traits, true>{
                c.data(), cast32u(c.size()),
                a.data(), cast32u(a.size())
        };
    }
    pcg::SchurVec<Traits, false> getParam() {
        return pcg::SchurVec<Traits, false>{
                c.data(), cast32u(c.size()),
                a.data(), cast32u(a.size())
        };
    }

    cudaError_t migrateToDevice(int deviceId, cudaStream_t stream) const;
#define USING_SCHUR_TYPE(type) using type = typename rba::Schur<Traits, false>::type
    USING_SCHUR_TYPE(EcBlock);
    USING_SCHUR_TYPE(EaBlock);
#undef USING_SCHUR_TYPE
    MngVector<EcBlock> c;
    MngVector<EaBlock> a;
};

template <typename Traits>
struct DevSchurDiag
{
    void clear() {M.clear(); U.clear();}
    void resize(uint32_t sizeM, uint32_t sizeU){
        clear();
        M.resize(sizeM);
        U.resize(sizeU);
    }
    pcg::SchurDiag<Traits, true> getParamConst() const {
        return pcg::SchurDiag<Traits, true>{
                M.data(), cast32u(M.size()),
                U.data(), cast32u(U.size())
        };
    }
    pcg::SchurDiag<Traits, false> getParam() {
        return pcg::SchurDiag<Traits, false>{
                M.data(), cast32u(M.size()),
                U.data(), cast32u(U.size())
        };
    }
    cudaError_t migrateToDevice(int deviceId, cudaStream_t stream) const;
#define USING_SCHUR_TYPE(type) using type = typename rba::Schur<Traits, false>::type
    USING_SCHUR_TYPE(MSymBlock);
    USING_SCHUR_TYPE(USymBlock);
#undef USING_SCHUR_TYPE
    MngVector<MSymBlock> M;
    MngVector<USymBlock> U;
};

template <typename Traits>
struct DevHessianVec
{
    void clear() {c.clear(); a.clear(); b.clear();}
    void resize(uint32_t sizeC, uint32_t sizeA, uint32_t sizeB){
        clear();
        c.resize(sizeC);
        a.resize(sizeA);
        b.resize(sizeB);
    }
    pcg::SchurVec<Traits, true> getSchurParamConst() const {
        return pcg::SchurVec<Traits, true>{
                c.data(), cast32u(c.size()),
                a.data(), cast32u(a.size())
        };
    }
    pcg::SchurVec<Traits, false> getSchurParam() {
        return pcg::SchurVec<Traits, false>{
                c.data(), cast32u(c.size()),
                a.data(), cast32u(a.size())
        };
    }
    HessianVec<Traits, true> getParamConst() const {
        return {
            c.data(), cast32u(c.size()),
            a.data(), cast32u(a.size()),
            b.data(), cast32u(b.size())
        };
    }
    HessianVec<Traits, false> getParam() {
        return {
            c.data(), cast32u(c.size()),
            a.data(), cast32u(a.size()),
            b.data(), cast32u(b.size())
        };
    }
    cudaError_t migrateToDevice(int deviceId, cudaStream_t stream) const;
#define USING_HESSIAN_TYPE(type) using type = typename rba::Hessian<Traits, false>::type
    USING_HESSIAN_TYPE(EcBlock);
    USING_HESSIAN_TYPE(EaBlock);
    USING_HESSIAN_TYPE(EbBlock);
#undef USING_HESSIAN_TYPE
    MngVector<EcBlock> c;
    MngVector<EaBlock> a;
    MngVector<EbBlock> b;
};

}//namespace grouped
}//namespace rba
