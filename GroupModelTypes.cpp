//
// Created by yao on 28/11/18.
//
#include "GroupModelTypes.h"
#include "GroupModel.h"
#include <unordered_map>
#include <boost/preprocessor/seq/for_each.hpp>
#include <iostream>

namespace rba{
namespace grouped{

template <typename Traits>
void DeviceModel<Traits>::init(const HostModel<Traits>& src) {
    points.resize(src.pointList.size());
    varPoints.clear(); varPoints.reserve(src.nbVarPoints);
    softCtrlPoints.idxVarPt.reserve(src.softCtrlPts.size());
    softCtrlPoints.data.reserve(src.softCtrlPts.size());
    for (uint32_t i = 0; i < src.pointList.size(); i++){
        uint32_t varIdx = rba::ModelBase<Traits>::varIdxFixed;
        if(src.pointVarMask[i]) {
            varIdx = uint32_t(varPoints.size());
            varPoints.push_back(i);
            if (src.softCtrlPts.count(i) != 0) {
                softCtrlPoints.idxVarPt.push_back(varIdx);
                assert(varPoints.at(varIdx) == i);
                softCtrlPoints.data.push_back(src.softCtrlPts.at(i));
            }
        }
        points[i] = {src.pointList[i], varIdx};
    }
    for (uint32_t i = 0; i < varPoints.size(); i++){
        assert(points[varPoints[i]].varIdx == i);
    }

    intrinsics.resize(src.intriList.size());
    varIntri.clear(); varIntri.reserve(src.nbVarIntri);
    for (uint32_t i = 0; i < src.intriList.size(); i++){
        uint32_t varIdx = rba::ModelBase<Traits>::varIdxFixed;
        if (src.intriVarMask[i]){
            varIdx = uint32_t(varIntri.size());
            varIntri.push_back(i);
        }
        intrinsics[i] = {src.intriList[i], varIdx};
    }
    for (uint32_t i = 0; i < varIntri.size(); i++){
        assert(intrinsics[varIntri[i]].varIdx == i);
    }

    captures.resize(src.capList.size());
    const bool hasGNSS = std::any_of(src.capList.begin(), src.capList.end(), [](const auto& cap){return cap.gnss.has_value();});
    if (hasGNSS) {
        gnssCtrl.resize(src.capList.size());
    }
    involvedCaps.clear();
    std::vector<uint32_t> fixedCaps;
    for (uint32_t i = 0; i < src.capList.size(); i++) {
        const auto& cap = src.capList[i];
        captures[i] = cap.capture;
        if (hasGNSS) {
            gnssCtrl.at(i) = cap.gnss.has_value() ? cap.gnss.value() : CtrlLoc<lpf>::makeDefault();
        }
        if (src.capVarMask[i]) {
            involvedCaps.push_back(i);
        }
        else {
            fixedCaps.push_back(i);
        }
    }
    nbVarCaps = uint32_t(src.nbVarCaps);
    assert(nbVarCaps == uint32_t(involvedCaps.size()));
    // Check if any of the fixed captures are involved. If yes, put them in involvedCaps.
    for (uint32_t idx : fixedCaps) {
        const auto& cap = src.capList[idx];
        const bool involvedByVarCam = isGroupModel<Traits>() ? src.intriVarMask[cap.capture.intrIdx] : false;
        const bool involvedByVarPt = std::any_of(cap.obs.begin(), cap.obs.end(), [&](const rba::CapOb<lpf>& ob)->bool{return src.pointVarMask[ob.ptIdx];});
        const bool involved = involvedByVarCam || involvedByVarPt;
        if (involved){
            involvedCaps.push_back(idx);
        }
    }

    capObs.reserveRows(src.capList.size());
    assert(src.nbTotalObs == std::accumulate(src.capList.begin(), src.capList.end(), size_t(0), [](size_t acc, const typename HostModel<Traits>::CaptureViews& a){return acc + a.obs.size();}));
    capObs.reserveData(std::accumulate(src.capList.begin(), src.capList.end(), size_t(0), [](size_t acc, const typename HostModel<Traits>::CaptureViews& a){return acc + roundUp(a.obs.size(), (size_t)decltype(capObs)::nbAlignedItems);}));
    for (unsigned i = 0; i < src.capList.size(); i++){
        const auto& obs = src.capList[i].obs;
        if(!std::is_sorted(obs.begin(), obs.end(), [](const rba::CapOb<lpf>& a, const rba::CapOb<lpf>& b){return a.ptIdx < b.ptIdx;})){
            throw std::runtime_error("observation list per capture shall be sorted");
        }
        capObs.appendRow(obs.data(), obs.size());
    }
}

template <typename Traits>
void DeviceModel<Traits>::saveToHost(HostModel<Traits>& dst) const {
    //@todo: make a utility function for this.
    const auto hostPts = points.getHostCopy();
    const auto hostIntrinsics = intrinsics.getHostCopy();
    const auto hostCaptures = captures.getHostCopy();

    assert(hostPts.size() == dst.pointList.size());
    for (uint32_t i = 0; i < dst.pointList.size(); i++){
        if(dst.pointVarMask[i]) {
            dst.pointList[i] = hostPts[i].pt;
        }
        else {
            assert(hostPts[i].varIdx == rba::ModelBase<Traits>::varIdxFixed);
            assert(dst.pointList[i] == hostPts[i].pt);
        }
    }

    assert(hostIntrinsics.size() == dst.intriList.size());
    for (uint32_t i = 0; i < dst.intriList.size(); i++){
        if (dst.intriVarMask[i]){
            dst.intriList[i] = hostIntrinsics[i].intri;
        }
        else {
            assert(hostIntrinsics[i].varIdx == rba::ModelBase<Traits>::varIdxFixed);
            assert(dst.intriList[i] == hostIntrinsics[i].intri);
        }
    }

    assert(hostCaptures.size() == dst.capList.size());
    for (uint32_t i = 0; i < dst.capList.size(); i++){
        if (dst.capVarMask[i]) {
            dst.capList[i].capture = hostCaptures[i];
        }
        else{
            assert(dst.capList[i].capture == hostCaptures[i]);
        }
    }
}

template <typename Traits>
void DeviceModel<Traits>::update(VarBackup& backup, const DevHessianVec<Traits>& delta, cudaStream_t stream){
    checkCudaErrors(updateModel::launchCudaUpdateModel(backup.template getParams<false>(), getKernelArgs<false>(), delta.getParamConst(), stream));
}
template <typename Traits>
void DeviceModel<Traits>::revert(const VarBackup& backup, cudaStream_t stream){
    checkCudaErrors(updateModel::launchCudaRevertModel(getKernelArgs<false>(), backup.getParams(), stream));
}

template <typename Traits>
cudaError_t DeviceModel<Traits>::migrateToDevice(int deviceId, cudaStream_t stream) const{
    checkEarlyReturn(intrinsics.migrateToDevice(deviceId, stream));
    checkEarlyReturn(varIntri.migrateToDevice(deviceId, stream));
    checkEarlyReturn(captures.migrateToDevice(deviceId, stream));
    checkEarlyReturn(gnssCtrl.migrateToDevice(deviceId, stream));
    checkEarlyReturn(involvedCaps.migrateToDevice(deviceId, stream));
    checkEarlyReturn(points.migrateToDevice(deviceId, stream));
    checkEarlyReturn(varPoints.migrateToDevice(deviceId, stream));
    checkEarlyReturn(softCtrlPoints.idxVarPt.migrateToDevice(deviceId, stream));
    checkEarlyReturn(softCtrlPoints.data.migrateToDevice(deviceId, stream));
    checkEarlyReturn(capObs.migrateToDevice(deviceId, stream));
    return cudaSuccess;
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template struct DeviceModel<TRAITS>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template <typename Traits>
void DeviceHessian<Traits>::init(const HostModel<Traits> &model, const rba::IdxCap *involvedCaps, size_t nbInvolvedCaps,
                         DeviceHessianPreCompData *preComp, bool computeHessianW) {
    using namespace rba;
    using lpf = typename Traits::lpf;
    constexpr auto varIdxFixed = rba::ModelBase<Traits>::varIdxFixed;

    const auto nbVarIntri = model.nbVarIntri;
    assert(nbVarIntri == size_t(std::count(model.intriVarMask.begin(), model.intriVarMask.end(), true)));
    M.resize(nbVarIntri);
    Ec.resize(nbVarIntri);

    const auto nbVarCaps = model.nbVarCaps;
    assert(nbVarCaps == size_t(std::count(model.capVarMask.begin(), model.capVarMask.end(), true)));
    U.resize(nbVarCaps);
    Ea.resize(nbVarCaps);
    Q.blocks.resize(nbVarCaps);
    Q.row.resize(nbVarCaps);
    std::fill(Q.row.begin(), Q.row.end(), badIdx<IdxMBlock>());
    // LUT for idxIntri -> idxMBlock. @todo: if only a small percentage of the intrinsics are variable, use a unordered_map instead and only store the variable ones
    std::vector<uint32_t> idxMMap(model.intriList.size());
    for (unsigned idxMBlock = 0, i = 0; i < model.intriList.size(); i++){
        idxMMap[i] = (model.intriVarMask[i] ? idxMBlock++ : rba::ModelBase<Traits>::varIdxFixed);
    }
    if (isGroupModel<Traits>()) {
        for (unsigned idxVarCap = 0, idxCap = 0; idxCap < model.capList.size(); idxCap++) {
            if (model.capVarMask[idxCap]) {
                const auto intriIdx = model.capList.at(idxCap).capture.intrIdx;
                Q.row[idxVarCap++] = idxMMap[intriIdx];
            }
        }
    }

    const auto nbVarPoints = model.nbVarPoints;
    V.resize(nbVarPoints);
    Eb.resize(nbVarPoints);

    if (preComp){
        preComp->nbVarCaps = uint32_t(nbVarCaps);
        preComp->nbInvolvedCaps = uint32_t(nbInvolvedCaps);
        preComp->rows.resize(nbInvolvedCaps + 1);
        preComp->rows[0] = 0;

        for (IdxUBlock idxInvolved = 0; idxInvolved < nbInvolvedCaps; idxInvolved++){
            const IdxCap idxCap = involvedCaps[idxInvolved];
            preComp->rows[idxInvolved + 1] = preComp->rows[idxInvolved] + DeviceHessianPreCompData::IdxObVarCap(
                    model.capList[idxCap].obs.size());
        }
        if (computeHessianW) {
            // sparseIdxW is required only when computing W
            preComp->sparseIdxW.resize(preComp->rows[preComp->nbVarCaps]);
        }
        preComp->sparseIdxS.resize(preComp->rows[preComp->nbInvolvedCaps]);

    }

    S.nbRows = IdxCam(nbVarIntri);
    S.rows.resize(S.nbRows + 1);
    S.rows[0] = 0;
    S.idxCol.clear();
    // LUT for idxPoint -> idxVBlock. @todo: if only a small percentage of the points are variable, use a unordered_map instead and only store the variable ones
    std::vector<IdxVBlock> idxVMap(model.pointList.size());
    for (unsigned idxV = 0, i = 0; i < model.pointList.size(); i++){
        idxVMap[i] = (model.pointVarMask[i] ? idxV++ : rba::ModelBase<Traits>::varIdxFixed);
    }
    // LUT for idxCap -> idxUBlock or index in involvedCaps. @todo: if only a small percentage of the points are variable, use a unordered_map instead and only store the variable ones
    std::vector<IdxUBlock> idxUMap(model.capList.size(), varIdxFixed);
    for (unsigned idxInvolved = 0; idxInvolved < nbInvolvedCaps; idxInvolved++){
        idxUMap[involvedCaps[idxInvolved]] = idxInvolved;
    }
    for (unsigned idxU = 0, i = 0; i < model.capList.size(); i++){
        if (model.capVarMask[i]){
            assert(idxUMap[i] == idxU);
            idxU++;
        }
        else{
            if(idxUMap[i] != rba::ModelBase<Traits>::varIdxFixed) {
                const auto iter = std::lower_bound(involvedCaps + nbVarCaps, involvedCaps + nbInvolvedCaps, i);
                assert(*iter == i && idxUMap[i] == iter - involvedCaps); unused(iter);
            }
        }
    }
    {
        // idxIntri of variable intrinsics to list of all involved captures (idxCap)
        const std::vector<std::pair<IdxCam, std::vector<IdxCap>>> camCaps = [&]() {
            std::unordered_map<IdxCam, std::vector<IdxCap>> camCapsMap{};
            std::vector<IdxCam> camIndices;
            if (isGroupModel<Traits>()) {
                for (uint32_t i = 0; i < nbInvolvedCaps; i++) {
                    const IdxCap idxCap = involvedCaps[i];
                    const IdxCam idxIntri = model.capList[idxCap].capture.intrIdx;
                    if (model.intriVarMask[idxIntri]) {
                        camCapsMap[idxIntri].push_back(idxCap);
                        if (camIndices.empty() || idxIntri !=
                                                  camIndices.back())// non-strict check to reduce mem size when all captures use the same camera.
                            camIndices.push_back(idxIntri);
                    }
                }
            }
            std::sort(camIndices.begin(), camIndices.end());
            camIndices.erase(std::unique(camIndices.begin(), camIndices.end()), camIndices.end());
            std::vector<std::pair<IdxCam, std::vector<IdxCap>>> result; result.reserve(camIndices.size());
            for(const auto& idxCam : camIndices){
                result.emplace_back(idxCam, std::move(camCapsMap.at(idxCam)));
            }
            return result;
        }();
        assert(camCaps.size() == nbVarIntri);

        // used for building preComp
        std::unordered_map<IdxVBlock, uint32_t> idxV2sparseIdxS;

        std::vector<IdxVBlock> camIdxVConcat;
        std::vector<uint32_t> ranges;
        std::vector<IdxVBlock> camIdxV;

        for (IdxMBlock idxM = 0; idxM < camCaps.size(); idxM++) {
            //const IdxCam idxIntri = camCaps[idxM].first;
            const std::vector<IdxCap> &caps = camCaps[idxM].second;
            camIdxVConcat.clear();
            ranges.clear();
            ranges.push_back(0);
            assert(ranges.size() == 1);
            camIdxV.clear();
            for (const auto& idxCap : caps){
                const auto &cap = model.capList[idxCap];
                for (const rba::CapOb<lpf> &ob : cap.obs){
                    if (model.pointVarMask[ob.ptIdx]){
                        camIdxVConcat.push_back(idxVMap[ob.ptIdx]);
                    }
                }
                assert(std::is_sorted(camIdxVConcat.begin() + ranges.back(), camIdxVConcat.end()));
                ranges.push_back(uint32_t(camIdxVConcat.size()));
            }
#if 1
            mergeUnique(camIdxV, &camIdxVConcat[0], ranges);
#else
            std::sort(camIdxV.begin(), camIdxV.end());
                camIdxV.erase(std::unique(camIdxV.begin(), camIdxV.end()), camIdxV.end());
#endif
            assert(S.idxCol.size() == S.rows[idxM]);
            S.idxCol.insert(S.idxCol.end(), camIdxV.begin(), camIdxV.end());
            S.rows[idxM+1] = uint32_t(S.idxCol.size());
            if (preComp) {//@fixme: also need to include sparseIdxS for fixed involved caps
                idxV2sparseIdxS.clear();
                for (uint32_t sparseIdxS = 0; sparseIdxS < camIdxV.size(); sparseIdxS++){
                    idxV2sparseIdxS.emplace(camIdxV[sparseIdxS], sparseIdxS);
                }
                for (const auto& idxCap : caps){
                    const auto& cap = model.capList[idxCap];
                    const IdxUBlock idxU = idxUMap[idxCap];
                    for (uint32_t obIdx = 0; obIdx < cap.obs.size(); obIdx++){
                        const uint32_t idxV = idxVMap[cap.obs[obIdx].ptIdx];
                        uint32_t sparseIdxS = rba::ModelBase<Traits>::varIdxFixed;
                        if (idxV != rba::ModelBase<Traits>::varIdxFixed) {
                            sparseIdxS = idxV2sparseIdxS.at(idxV);
                            assert(camIdxV[sparseIdxS] == idxV);
                        }
                        preComp->sparseIdxS[preComp->rows[idxU] + obIdx] = sparseIdxS;
                    }
                }
            }
        }
        assert(camCaps.size() == nbVarIntri);
        S.data.resize(S.rows[nbVarIntri]);

    }

    // Do we need sparse pattern/shape of W when computeHessianW=false? seems yes.
//    if (computeHessianW)
    {

        W.nbRows = uint32_t(model.nbVarCaps);
        W.rows.resize(W.nbRows + 1);
        W.rows[0] = 0;
        W.idxCol.clear();
        {
            std::vector<IdxVBlock> capIdxV;
            IdxUBlock idxU = 0;
            for (IdxCap idxCap = 0; idxCap < model.capList.size(); idxCap++) {
                if (model.capVarMask[idxCap]) {
                    capIdxV.clear();
                    const auto &cap = model.capList[idxCap];
                    for (const rba::CapOb<lpf> &ob : cap.obs) {
                        if (model.pointVarMask[ob.ptIdx]) {
                            const IdxVBlock idxV = idxVMap[ob.ptIdx];
                            capIdxV.push_back(idxV);
                        }
                    }
                    assert(std::is_sorted(capIdxV.begin(), capIdxV.end()));
                    assert(idxU == idxUMap[idxCap]);
                    W.rows[idxU + 1] = W.rows[idxU] + uint32_t(capIdxV.size());
                    W.idxCol.insert(W.idxCol.end(), capIdxV.begin(), capIdxV.end());
                    // pre-compute
                    if (preComp && computeHessianW) {
                        auto sparseIdxWLast = std::numeric_limits<uint32_t>::max();// defined overflow to zero for the first iteration.
                        for (uint32_t i = 0; i < cap.obs.size(); i++) {
                            const IdxVBlock idxV = idxVMap[cap.obs[i].ptIdx];
                            uint32_t sparseIdxW = sparseIdxWLast + 1;
                            if (idxV != ModelBase<Traits>::varIdxFixed) {
                                while (capIdxV[sparseIdxW] < idxV) {
                                    sparseIdxW++;
                                    assert(sparseIdxW < capIdxV.size());
                                }
                                assert(capIdxV[sparseIdxW] == idxV);
                                sparseIdxWLast = sparseIdxW;
                            } else {
                                sparseIdxW = ModelBase<Traits>::varIdxFixed;
                            }
                            preComp->sparseIdxW[preComp->rows[idxU] + i] = sparseIdxW;
                            assert(preComp->rows[idxU] + cap.obs.size() == preComp->rows[idxU + 1]);
                        }
                    }
                    idxU++;
                }
            }
        }
    }
    if (computeHessianW) {
        W.data.resize(W.rows[W.nbRows]);
    }
}
template <typename Traits>
cudaError_t DeviceHessian<Traits>::migrateToDevice(int deviceId, cudaStream_t stream) const{
    checkEarlyReturn(M.migrateToDevice(deviceId, stream));
    checkEarlyReturn(U.migrateToDevice(deviceId, stream));
    checkEarlyReturn(V.migrateToDevice(deviceId, stream));
    checkEarlyReturn(Q.blocks.migrateToDevice(deviceId, stream));
    checkEarlyReturn(Q.row.migrateToDevice(deviceId, stream));
    checkEarlyReturn(S.migrateToDevice(deviceId, stream));
    checkEarlyReturn(W.migrateToDevice(deviceId, stream));
    checkEarlyReturn(Ec.migrateToDevice(deviceId, stream));
    checkEarlyReturn(Ea.migrateToDevice(deviceId, stream));
    checkEarlyReturn(Eb.migrateToDevice(deviceId, stream));
    return cudaSuccess;
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template struct DeviceHessian<TRAITS>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template <typename Traits>
std::vector<std::vector<uint16_t>> make_sparseIdxW2idxLocalOb(const DeviceModel<Traits> &model, const DeviceHessian<Traits> &hessian){
    const auto nbVarCaps = hessian.getKernelArgsConst().nbVarCaps;
    std::vector<std::vector<uint16_t>> sparseIdxW2IdxLocalOb(nbVarCaps);
    for (IdxUBlock i = 0; i < nbVarCaps; i++){
        const IdxCap idxCap = model.involvedCaps[i];
        const auto obs = model.capObs.row(idxCap);;
        const auto nbObs = numeric_cast<uint32_t>(model.capObs.rowSize(idxCap));
        const IdxVBlock* pIdxV = &hessian.W.idxCol[hessian.W.rows[i]];
        for (unsigned j = 0; j < nbObs; j++){
            const auto& ob = obs[j];
            const IdxVBlock idxV = model.points[ob.ptIdx].varIdx;
            if (idxV != ModelBase<Traits>::varIdxFixed){
                assert(pIdxV[sparseIdxW2IdxLocalOb[i].size()] == idxV); unused(pIdxV);
                sparseIdxW2IdxLocalOb[i].push_back(numeric_cast<uint16_t>(j));
            }
        }
    }
    return sparseIdxW2IdxLocalOb;
}
template <typename Traits>
void DeviceSchur<Traits>::init(const DeviceModel<Traits> &model, const DeviceHessian<Traits> &hessian, DeviceSchurPreCompData *precomp,
                       std::shared_ptr<DevVecVec<uint16_t>> pSparseIdxW2idxLocalOb) {
    clear();
    using namespace rba;
    const uint32_t nbVarIntri = cast32u(model.varIntri.size()); assert(nbVarIntri == hessian.M.size());
    const uint32_t nbVarCaps = model.nbVarCaps; assert(model.nbVarCaps == hessian.U.size());

    diagM.resize(nbVarIntri);
    Ec.resize(nbVarIntri);

    struct Coord2D{
        uint32_t row;//IdxMBlock or IdxUBlock
        IdxVBlock col;
    };
    struct ColGreater{
        bool operator()(const Coord2D& a, const Coord2D& b) {
            auto make64u = [](const Coord2D& src)->uint64_t {return (uint64_t(src.col) << 32) | uint64_t(src.row); };
            return make64u(a) > make64u(b);
        }
    };
    using SparseIdxS = uint32_t;
    using SparseIdxW = uint32_t;
    std::vector<std::unordered_map<IdxMBlock, std::vector<std::pair<SparseIdxS, SparseIdxS>>>> preCompUpperM(nbVarIntri);
    std::vector<std::unordered_map<IdxUBlock, std::vector<std::pair<SparseIdxW, SparseIdxW>>>> preCompUpperU(nbVarCaps);
    std::vector<std::unordered_map<IdxMBlock, std::vector<std::pair<SparseIdxS, SparseIdxW>>>> preCompQ(nbVarIntri);

    // Put hessian.Q in, so schur.Q will always cover hessian.Q, even if a capture only observes fixed points
    for (IdxMBlock col = 0; col < nbVarCaps; col++){
        const auto row = hessian.Q.row[col];
        if (row != ModelBase<Traits>::varIdxFixed){
            preCompQ[row][col] = {};
        }
    }

    if constexpr (false)
    {// this implementation is not always faster and consumes more memory but maybe more easily parallelized
        // sparse pattern for S and W
        struct PatternSW{
            uint32_t nbRows;
            IdxVBlock nbCols;
            const uint32_t* rows; // length = nbRows + 1, rows[0] = 0
            const IdxVBlock* idxCol; //col index, length = rows[nbRows]
        };
        const auto makeRowsPerCol = [](const PatternSW& csr){
            std::vector<std::vector<uint32_t>> rowsPerCol(csr.nbCols);
            for (uint32_t idxRow = 0; idxRow < csr.nbRows; idxRow++){
                // This sparseIdxCol is different from sparseIdxCol used elsewhere in that it is based from idxCols[0], not idxCols[rows[idxRow]]
                for (auto sparseIdxCol = csr.rows[idxRow]; sparseIdxCol < csr.rows[idxRow + 1]; sparseIdxCol++){
                    const uint32_t idxCol = csr.idxCol[sparseIdxCol];
                    rowsPerCol[idxCol].push_back(idxRow);
                }
            }
            for (auto& r : rowsPerCol){
                std::sort(r.begin(), r.end());
            }
            return rowsPerCol;
        };
        const IdxVBlock nbVarPts = hessian.V.size();
        const PatternSW patternS {nbVarIntri, nbVarPts, hessian.S.rows.data(), hessian.S.idxCol.data()};
        const PatternSW patternW {nbVarCaps, nbVarPts, hessian.W.rows.data(), hessian.W.idxCol.data()};
        const std::vector<std::vector<IdxMBlock>> rowsPerSCol = makeRowsPerCol(patternS);
        const std::vector<std::vector<IdxUBlock>> rowsPerWCol = makeRowsPerCol(patternW);

        const auto findSparseIdxForCol = [](const PatternSW& csr, IdxVBlock idxCol){
            using SparseIdx = uint32_t;
            std::vector<SparseIdx> result(csr.nbRows, 0u);
            for (size_t i = 0; i < csr.nbRows; i++){
                const auto iter = std::lower_bound(&csr.idxCol[csr.rows[i]], &csr.idxCol[csr.rows[i+1]], idxCol);
                result[i] = cast32u(iter - &csr.idxCol[csr.rows[i]]);
            }
            return result;
        };
        const auto makePreCompUpperUMQ = [&patternS, &patternW, &rowsPerSCol, &rowsPerWCol, &findSparseIdxForCol,
                                          &preCompUpperM, &preCompUpperU, &preCompQ]
                (IdxVBlock colBeg, IdxVBlock colEnd){
            // Current sparse column index per row
            std::vector<SparseIdxS> sparseIdxS = findSparseIdxForCol(patternS, colBeg);
            std::vector<SparseIdxW> sparseIdxW = findSparseIdxForCol(patternW, colBeg);
            for (IdxVBlock idxCol = colBeg; idxCol < colEnd; idxCol++) {
                const auto& idxRowsS = rowsPerSCol[idxCol];
                const auto& idxRowsW = rowsPerWCol[idxCol];
                const auto updatePreCompUpperMU =
                        [](std::vector<std::unordered_map<IdxMBlock, std::vector<std::pair<SparseIdxS, SparseIdxS>>>>& preCompUpperMU,
                            const std::vector<uint32_t>& idxRows, const std::vector<uint32_t>& sparseIdx, const uint32_t nbRows)
                {
                    assert(preCompUpperMU.size() == nbRows);
                    assert(std::is_sorted(idxRows.begin(), idxRows.end()));
                    for (unsigned i = 0; i < idxRows.size(); i++) {
                        const IdxMBlock rowA = idxRows[i];
                        for (unsigned j = i + 1; j < idxRows.size(); j++) {
                            const IdxMBlock rowB = idxRows[j];
                            preCompUpperMU[rowA][rowB].push_back({sparseIdx[rowA], sparseIdx[rowB]});
                        }
                    }
                };
                updatePreCompUpperMU(preCompUpperM, idxRowsS, sparseIdxS, patternS.nbRows);
                for (auto idxRow : idxRowsS) { sparseIdxS[idxRow]++; }
                updatePreCompUpperMU(preCompUpperU, idxRowsW, sparseIdxW, patternW.nbRows);
                for (auto idxRow : idxRowsW) { sparseIdxW[idxRow]++; }

                for (const IdxMBlock rowA : idxRowsS) {
                    for (const IdxUBlock rowB : idxRowsW) {
                        preCompQ[rowA][rowB].push_back({sparseIdxS[rowA] - 1, sparseIdxW[rowB] - 1});
                    }
                }
            }
        };
        makePreCompUpperUMQ(0u, nbVarPts);
    }
    else
    {
        std::priority_queue<Coord2D, std::vector<Coord2D>, ColGreater> heapS;
        std::priority_queue<Coord2D, std::vector<Coord2D>, ColGreater> heapW;
        std::vector<SparseIdxS> sparseIdxS(nbVarIntri, 0u);
        assert(sparseIdxS.size() == model.varIntri.size());
        auto heapSItem = [&](IdxMBlock row) {
            if (sparseIdxS[row] < hessian.S.rows[row + 1] - hessian.S.rows[row])
                heapS.push(Coord2D{row, hessian.S.idxCol[hessian.S.rows[row] + sparseIdxS[row]++]});
        };
        std::vector<SparseIdxW> sparseIdxW(nbVarCaps, 0u);
        assert(sparseIdxW.size() == model.nbVarCaps);
        auto heapWItem = [&](IdxUBlock row) {
            if (sparseIdxW[row] < hessian.W.rows[row + 1] - hessian.W.rows[row])
                heapW.push(Coord2D{row, hessian.W.idxCol[hessian.W.rows[row] + sparseIdxW[row]++]});
        };
        for (IdxMBlock i = 0; i < nbVarIntri; i++)
            heapSItem(i);
        for (IdxUBlock i = 0; i < nbVarCaps; i++)
            heapWItem(i);
        std::vector<IdxMBlock> rowS; //!< list of idxRow for a specific idxCol
        std::vector<IdxUBlock> rowW; //!< list of idxRow for a specific idxCol
        while (!heapS.empty() || !heapW.empty()) {
            rowS.clear();
            rowW.clear();
            const IdxVBlock col = std::min(
                    heapS.empty() ? std::numeric_limits<IdxVBlock>::max() : heapS.top().col,
                    heapW.empty() ? std::numeric_limits<IdxVBlock>::max() : heapW.top().col);
            while (!heapS.empty() && heapS.top().col == col) {
                const IdxMBlock row = heapS.top().row;
                rowS.push_back(row);
                heapS.pop();
            }
            assert(std::is_sorted(rowS.begin(), rowS.end()));
            for (unsigned i = 0; i < rowS.size(); i++) {
                const IdxMBlock rowA = rowS[i];
                for (unsigned j = i + 1; j < rowS.size(); j++) {
                    const IdxMBlock rowB = rowS[j];
                    preCompUpperM[rowA][rowB].push_back({sparseIdxS[rowA] - 1, sparseIdxS[rowB] - 1});
                }
            }

            while (!heapW.empty() && heapW.top().col == col) {
                const IdxUBlock row = heapW.top().row;
                rowW.push_back(row);
                heapW.pop();
            }
            assert(std::is_sorted(rowW.begin(), rowW.end()));
            for (int i = 0; i < int(rowW.size()); i++) {
                const IdxUBlock rowA = rowW[i];
                for (unsigned j = i + 1; j < rowW.size(); j++) {
                    const IdxUBlock rowB = rowW[j];
                    preCompUpperU[rowA][rowB].push_back({sparseIdxW[rowA] - 1, sparseIdxW[rowB] - 1});
                }
            }

            for (const IdxMBlock rowA : rowS) {
                for (const IdxUBlock rowB : rowW) {
                    preCompQ[rowA][rowB].push_back({sparseIdxS[rowA] - 1, sparseIdxW[rowB] - 1});
                }
            }

            for (IdxMBlock row : rowS)
                heapSItem(row);
            for (IdxUBlock row : rowW)
                heapWItem(row);
        }
    }
    upperM.nbRows = nbVarIntri;
    upperM.rows.resize(nbVarIntri + 1);
    upperM.rows[0] = 0;
    for(unsigned i = 0; i < nbVarIntri; i++){
        upperM.rows[i+1] = numeric_cast<uint32_t>(upperM.rows[i] + preCompUpperM[i].size());
    }
    upperM.idxCol.resize(upperM.rows.back());
    upperM.data.resize(upperM.rows.back());
    for (IdxMBlock i = 0; i < nbVarIntri; i++) {
        auto pBeg = &upperM.idxCol[upperM.rows[i]];
        auto pEnd = &upperM.idxCol[upperM.rows[i+1]];
        const auto& row = preCompUpperM[i];
        assert(ptrdiff_t(row.size()) == pEnd - pBeg);
        std::transform(row.begin(), row.end(), pBeg, [](const std::pair<IdxMBlock, std::vector<std::pair<SparseIdxS, SparseIdxS>>>& item){return item.first;});
        std::sort(pBeg, pEnd);
    }

    if (precomp) {
        precomp->pairSS.clear();
        using PairSS = typename rba::SchurUpperMComputer::PreComp::PairSS;
        std::vector<PairSS> pairs;
        for (IdxMBlock i = 0; i < nbVarIntri; i++) {
            const auto &row = preCompUpperM[i];
            for (uint32_t j = upperM.rows[i]; j < upperM.rows[i + 1]; j++) {
                pairs.clear();
                const auto& src = row.at(upperM.idxCol[j]);
                std::transform(src.begin(), src.end(), std::back_inserter(pairs),
                               [](const std::pair<SparseIdxS, SparseIdxS>& p)->PairSS{return {p.first, p.second};});
                assert(std::is_sorted(pairs.begin(), pairs.end(), [](const PairSS& p0, const PairSS& p1){return p0.a < p1.a;}));
                assert(std::is_sorted(pairs.begin(), pairs.end(), [](const PairSS& p0, const PairSS& p1){return p0.b < p1.b;}));
                precomp->pairSS.appendRow(pairs.data(), numeric_cast<uint32_t>(pairs.size()));
            }
        }
    }

    const std::vector<std::vector<uint16_t>> sparseIdxW2IdxLocalOb =
            pSparseIdxW2idxLocalOb != nullptr
            ? toHost_sparseIdxW2idxLocalOb(pSparseIdxW2idxLocalOb->getHostCopy())
            : make_sparseIdxW2idxLocalOb(model, hessian);
    if (precomp){
        if (pSparseIdxW2idxLocalOb != nullptr)
            precomp->sparseIdxW2idxLocalOb = pSparseIdxW2idxLocalOb;
        else {
            if (precomp->sparseIdxW2idxLocalOb == nullptr ) {
                precomp->sparseIdxW2idxLocalOb = std::make_shared<DevVecVec<uint16_t>>();
            }
            toDevice_sparseIdxW2idxLocalOb(*precomp->sparseIdxW2idxLocalOb, sparseIdxW2IdxLocalOb);
        }
    }

    diagU.resize(nbVarCaps);
    Ea.resize(nbVarCaps);

    upperU.nbRows = nbVarCaps;
    upperU.rows.resize(nbVarCaps + 1);
    upperU.rows[0] = 0;
    for (uint32_t i = 0; i < nbVarCaps; i++)
        upperU.rows[i+1] = numeric_cast<uint32_t>(upperU.rows[i] + preCompUpperU[i].size());
    upperU.idxCol.resize(upperU.rows.back());
    upperU.data.resize(upperU.rows.back());
    for (IdxUBlock i = 0; i < nbVarCaps; i++) {
        auto pBeg = &upperU.idxCol[upperU.rows[i]];
        auto pEnd = &upperU.idxCol[upperU.rows[i+1]];
        const auto& row = preCompUpperU[i];
        assert(ptrdiff_t(row.size()) == pEnd - pBeg);
        std::transform(row.begin(), row.end(), pBeg, [](const std::pair<IdxUBlock, std::vector<std::pair<SparseIdxW, SparseIdxW>>>& item){return item.first;});
        std::sort(pBeg, pEnd);
    }

    if (precomp){
        precomp->pairWW.clear();
        using Pair = rba::SchurUpperUComputer::ObPair;
        std::vector<Pair> pairs;
        for (IdxUBlock i = 0; i < nbVarCaps; i++) {
            const auto& row = preCompUpperU[i];
            for (uint32_t j = upperU.rows[i]; j < upperU.rows[i+1]; j++){
                pairs.clear();
                const auto& src = row.at(upperU.idxCol[j]);
                std::transform(src.begin(), src.end(), std::back_inserter(pairs),
                               [&](const std::pair<SparseIdxW, SparseIdxW>& p)->Pair{
                                   assert(hessian.W.idxCol[hessian.W.rows[i] + p.first] == hessian.W.idxCol[hessian.W.rows[upperU.idxCol[j]] + p.second]);
                                   const Pair pair = {sparseIdxW2IdxLocalOb[i][p.first], sparseIdxW2IdxLocalOb[upperU.idxCol[j]][p.second]};
                                   assert(model.capObs.row(model.involvedCaps[i])[pair.idxLocalObA].ptIdx ==  model.capObs.row(model.involvedCaps[upperU.idxCol[j]])[pair.idxLocalObB].ptIdx);
                                   return pair;
                               });
                assert(std::is_sorted(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b){return a.idxLocalObA < b.idxLocalObA;}));
                assert(std::is_sorted(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b){return a.idxLocalObB < b.idxLocalObB;}));
                precomp->pairWW.appendRow(pairs.data(), numeric_cast<uint32_t>(pairs.size()));
            }
        }
    }

    Q.nbRows = nbVarIntri;
    Q.rows.resize(nbVarIntri + 1);
    Q.rows[0] = 0;
    for (uint32_t i = 0; i < nbVarIntri; i++)
        Q.rows[i+1] = numeric_cast<uint32_t>(Q.rows[i] + preCompQ[i].size());
    Q.idxCol.resize(Q.rows.back());
    Q.data.resize(Q.rows.back());
    for (IdxMBlock i = 0; i < nbVarIntri; i++){
        auto pBeg = &Q.idxCol[Q.rows[i]];
        auto pEnd = &Q.idxCol[Q.rows[i+1]];
        const auto& row = preCompQ[i];
        assert(ptrdiff_t(row.size()) == pEnd - pBeg);
        std::transform(row.begin(), row.end(), pBeg, [](const std::pair<IdxUBlock, std::vector<std::pair<SparseIdxS, SparseIdxW>>>& item){return item.first;});
        std::sort(pBeg, pEnd);
    }
#ifndef NDEBUG
    for (auto& item : Q.data)
        std::fill_n(item.data(), item.size(), NAN);
#endif
    if (precomp){
        precomp->qPairIdxLocalOb.clear();
        precomp->qPairSparseIdxS.clear();
        struct Pair{
            uint32_t sparseIdxS;
            uint16_t idxLocalOb;
        };
        std::vector<uint32_t> sparseIdxS;
        std::vector<uint16_t> idxLocalOb;
        for (IdxMBlock i = 0; i < nbVarIntri; i++){
            const auto& row = preCompQ[i];
            for (uint32_t j = Q.rows[i]; j < Q.rows[i+1]; j++){
                sparseIdxS.clear();
                idxLocalOb.clear();
                const auto& src = row.at(Q.idxCol[j]);
                for (const auto& item : src){
                    sparseIdxS.push_back(item.first);
                    idxLocalOb.push_back(sparseIdxW2IdxLocalOb[Q.idxCol[j]][item.second]);
                }
                assert(std::is_sorted(sparseIdxS.begin(), sparseIdxS.end()));
                assert(std::is_sorted(idxLocalOb.begin(), idxLocalOb.end()));
                precomp->qPairSparseIdxS.appendRow(sparseIdxS.data(), numeric_cast<uint32_t>(sparseIdxS.size()));
                precomp->qPairIdxLocalOb.appendRow(idxLocalOb.data(), numeric_cast<uint32_t>(idxLocalOb.size()));
            }
        }
    }
    //check schur.Q covers hessian.Q
    for (IdxUBlock i = 0; i < nbVarCaps; i++){
        const auto row = hessian.Q.row[i];
        if (row != ModelBase<Traits>::varIdxFixed){
            assert(std::find(&Q.idxCol[Q.rows[row]], &Q.idxCol[Q.rows[row+1]], i) != &Q.idxCol[Q.rows[row+1]]);
        }
    }

    if (precomp){
        precomp->rowTableUpperM.resize(upperM.idxCol.size());
        for (IdxMBlock i = 0; i < nbVarIntri; i++){
            std::fill_n(&precomp->rowTableUpperM[upperM.rows[i]], upperM.rows[i+1]-upperM.rows[i], i);
        }
        precomp->rowTableUpperU.resize(upperU.idxCol.size());
        for (IdxUBlock i = 0; i < nbVarCaps; i++){
            std::fill_n(&precomp->rowTableUpperU[upperU.rows[i]], upperU.rows[i+1]-upperU.rows[i], i);
        }
        precomp->rowTableQ.resize(Q.idxCol.size());
        for (IdxMBlock i = 0; i < nbVarIntri; i++){
            std::fill_n(&precomp->rowTableQ[Q.rows[i]], Q.rows[i+1] - Q.rows[i], i);
        }
    }
}



void toDevice_sparseIdxW2idxLocalOb(DevVecVec<uint16_t> &dst, const std::vector<std::vector<uint16_t>> &src)
{
    dst.clear();
    for (const auto& row : src){
        dst.appendRow(row.data(), numeric_cast<uint32_t>(row.size()));
    }
}
std::vector<std::vector<uint16_t>> toHost_sparseIdxW2idxLocalOb(const DevVecVec<uint16_t>& src){
    std::vector<std::vector<uint16_t>> dst;
    dst.reserve(src.rows());
    for (uint32_t i = 0; i < src.rows(); i++){
        dst.emplace_back(src.row(i), src.row(i) + src.rowSize(i));
    }
    return dst;
}

template <typename Traits>
void DevicePCGPreComp::init(const rba::grouped::DeviceSchur<Traits> &devSchur) {
    const auto nbVarIntri = rba::IdxMBlock(devSchur.diagM.size());
    const auto nbVarCaps = rba::IdxUBlock(devSchur.diagU.size());
    transposeCSR(lowerM, devSchur.upperM, nbVarIntri);
    transposeCSR(lowerU, devSchur.upperU, nbVarCaps);
    transposeCSR(lowerQ, devSchur.Q, nbVarCaps);
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template void DevicePCGPreComp::init(const rba::grouped::DeviceSchur<TRAITS> &schur);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

cudaError_t DeviceHessianPreCompData::migrateToDevice(int deviceId, cudaStream_t stream) const{
    checkEarlyReturn(rows.migrateToDevice(deviceId, stream));
    checkEarlyReturn(sparseIdxS.migrateToDevice(deviceId, stream));
    checkEarlyReturn(sparseIdxW.migrateToDevice(deviceId, stream));
    return cudaSuccess;
}

template <typename Traits>
cudaError_t DeviceSchur<Traits>::migrateToDevice(int deviceId, cudaStream_t stream) const{
    checkEarlyReturn(upperM.migrateToDevice(deviceId, stream));
    checkEarlyReturn(diagM.migrateToDevice(deviceId, stream));
    checkEarlyReturn(Ec.migrateToDevice(deviceId, stream));
    checkEarlyReturn(upperU.migrateToDevice(deviceId, stream));
    checkEarlyReturn(diagU.migrateToDevice(deviceId, stream));
    checkEarlyReturn(Ea.migrateToDevice(deviceId, stream));
    checkEarlyReturn(Q.migrateToDevice(deviceId, stream));
    return cudaSuccess;
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template struct DeviceSchur<TRAITS>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

cudaError_t DeviceSchurPreCompData::migrateToDevice(int deviceId, cudaStream_t stream) const{
    checkEarlyReturn(rowTableUpperM.migrateToDevice(deviceId, stream));
    checkEarlyReturn(rowTableUpperU.migrateToDevice(deviceId, stream));
    checkEarlyReturn(pairSS.migrateToDevice(deviceId, stream));
    checkEarlyReturn(pairWW.migrateToDevice(deviceId, stream));
    checkEarlyReturn(sparseIdxW2idxLocalOb->migrateToDevice(deviceId, stream));
    checkEarlyReturn(qPairSparseIdxS.migrateToDevice(deviceId, stream));
    checkEarlyReturn(qPairIdxLocalOb.migrateToDevice(deviceId, stream));
    checkEarlyReturn(rowTableQ.migrateToDevice(deviceId, stream));
    return cudaSuccess;
}

cudaError_t DevicePCGPreComp::migrateToDevice(int deviceId, cudaStream_t stream) const{
    checkEarlyReturn(lowerM.migrateToDevice(deviceId, stream));
    checkEarlyReturn(lowerU.migrateToDevice(deviceId, stream));
    checkEarlyReturn(lowerQ.migrateToDevice(deviceId, stream));
    return cudaSuccess;
}

template <typename Traits>
cudaError_t DevSchurVec<Traits>::migrateToDevice(int deviceId, cudaStream_t stream) const{
    checkEarlyReturn(c.migrateToDevice(deviceId, stream));
    checkEarlyReturn(a.migrateToDevice(deviceId, stream));
    return cudaSuccess;
}
template <typename Traits>
cudaError_t DevSchurDiag<Traits>::migrateToDevice(int deviceId, cudaStream_t stream) const{
    checkEarlyReturn(M.migrateToDevice(deviceId, stream));
    checkEarlyReturn(U.migrateToDevice(deviceId, stream));
    return cudaSuccess;
}
template <typename Traits>
cudaError_t DevHessianVec<Traits>::migrateToDevice(int deviceId, cudaStream_t stream) const{
    checkEarlyReturn(a.migrateToDevice(deviceId, stream));
    checkEarlyReturn(b.migrateToDevice(deviceId, stream));
    checkEarlyReturn(c.migrateToDevice(deviceId, stream));
    return cudaSuccess;
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template struct DevHessianVec<TRAITS>;\
    template struct DevSchurDiag<TRAITS>;\
    template struct DevSchurVec<TRAITS>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES
}//namespace grouped
}//namespace rba