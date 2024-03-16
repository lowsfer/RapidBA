//
// Created by yao on 9/20/19.
//

#include <boost/preprocessor/seq.hpp>
#include <boost/preprocessor/cat.hpp>
#include <unordered_set>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/StdVector>
#include <eigen3/Eigen/SVD>
#include "traits/Traits.h"
#include "ModelFilter.h"
#include <iostream>
#include <boost/format.hpp>
using format = boost::format;
namespace rba::grouped {

template <typename Traits>
bool ModelFilter<Traits>::isValidCam(const IdxCam idxCam) const {
    assert(mSrc.intriVarMask.at(idxCam));
    if (mCamCaps.at(idxCam).empty()){
        return false;
    }
    return true;
}
template <typename Traits>
bool ModelFilter<Traits>::isValidOb(const IdxCap idxCap, const ModelFilter<Traits>::IdxLocalOb idxOb) const {
    const auto& cap = mSrc.capList.at(idxCap);
    const auto& ob = cap.obs[idxOb];
    const auto& cam = isGroupModel<Traits>() ? mSrc.intriList.at(cap.capture.intrIdx) : typename Traits::CamIntr{};
    const auto& intrinsics = getIntrinsics(cam, cap.capture);
    Eigen::Vector2f c, f, fInv;
    c << float(intrinsics.cx()), float(intrinsics.cy());
    f << float(intrinsics.fx()), float(intrinsics.fy());
    fInv = 1.f / f.array();
    const Eigen::Vector2f normXY = Eigen::Vector2f::MapAligned(ob.position).array() * fInv.array();
    if (normXY.squaredNorm() > sqr(mMaxCapObTanAngle)){
        return false;
    }

    using lpf = typename Traits::lpf;
    using locf = typename Traits::locf;
    const kmat<lpf, 3> transPt = quat2mat(kmat<lpf, 4>{cap.capture.pose.q}) * (kmat<locf, 3>{mSrc.pointList[ob.ptIdx].position} - kmat<locf, 3>{cap.capture.pose.c});
    if (transPt.sqrNorm() < sqr(mRefDistance * mMinCapPtRelativeDistance)) {
        return false;
    }
    return true;
}

template <typename Traits>
bool ModelFilter<Traits>::isValidCap(rba::IdxCap idxCap) const {
    assert(mSrc.capVarMask.at(idxCap));

    const auto& cap = mSrc.capList.at(idxCap);
    if (isGroupModel<Traits>()){
        if (!mCamValidMask.at(cap.capture.intrIdx)){
            return false;
        }
    }

    const auto& obsValidMask = mCapObsValidMask.at(idxCap);
    const size_t nbValidObs = std::count(obsValidMask.begin(), obsValidMask.end(), true);
    
    if (cap.gnss.has_value()) {
        return nbValidObs * 2 >= mMinCapObs * 2 - 3;
    }

    if (nbValidObs < mMinCapObs){
        return false;
    }

    const auto& cam = isGroupModel<Traits>() ? mSrc.intriList.at(cap.capture.intrIdx) : typename Traits::CamIntr{};
    const auto& intrinsics = getIntrinsics(cam, cap.capture);
    Eigen::Matrix2f normXYCovariance;
    {
        Eigen::Matrix2f a = Eigen::Matrix2f::Zero();
        Eigen::Vector2f b = Eigen::Vector2f::Zero();
        Eigen::Vector2f c, f, fInv;
        c << float(intrinsics.cx()), float(intrinsics.cy());
        f << float(intrinsics.fx()), float(intrinsics.fy());
        fInv = 1.f / f.array();
        for (size_t i = 0; i < cap.obs.size(); i++) {
            if (obsValidMask[i]) {
                // replacing (proj-c) with proj should make no difference
                const Eigen::Vector2f normXY = Eigen::Vector2f::MapAligned(cap.obs[i].position).array() * fInv.array();
//                const Eigen::Vector2f normXY = (Eigen::Vector2f::MapAligned(cap.obs[i].position) - c).array() * fInv.array();
                a += normXY * normXY.transpose();
                b += normXY;
            }
        }
        const float scale = 1.f/float(nbValidObs - 1);
        normXYCovariance = a * scale - sqr(scale) * b * b.transpose();
    }
    Eigen::JacobiSVD svd(normXYCovariance);
    const Eigen::Vector2f& singularValues = svd.singularValues();
    if (singularValues.minCoeff() < sqr(mMinCapObObSinAngle)){
        return false;
    }
    return true;
}

template <typename Traits>
bool ModelFilter<Traits>::isValidPt(rba::IdxPt idxPt) const{
    assert(mSrc.pointVarMask.at(idxPt));
    if (mSrc.softCtrlPts.count(idxPt) != 0) {
        return true;
    }
    using lpf = typename Traits::lpf;
    using locf = typename Traits::locf;
    const kmat<locf, 3> pt{mSrc.pointList.at(idxPt).position};
    const auto& obs = mPtObs.at(idxPt);
    // check number of captures observing the point
    const size_t nbValidObs = std::count_if(obs.begin(), obs.end(), [this](const PtOb& ob) { return mCapObsValidMask[ob.idxCap][ob.idxOb]; });
    if (nbValidObs < mMinPtObs){
        return false;
    }
    // check if the point can be triangulated effectively
    {
        thread_local std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> normalizedRays;
        normalizedRays.clear();
        for (const auto& ob : obs) {
            if (mCapObsValidMask[ob.idxCap][ob.idxOb]) {
                assert(mCapValidMask[ob.idxCap]);
                const auto& capPose = mSrc.capList.at(ob.idxCap).capture.pose;
                const kmat<lpf, 3> transPt = quat2mat(kmat<lpf, 4>{capPose.q}) * (pt - kmat<locf, 3>{capPose.c});
                normalizedRays.push_back(Eigen::Vector3f::Map(transPt.data()).normalized());
            }
        }
        bool isValid = false;
        for (size_t i = 0; i < normalizedRays.size(); i++) {
            for (size_t j = i + 1; j < normalizedRays.size(); j++) {
                if (std::abs(normalizedRays[i].dot(normalizedRays[j])) < mMaxPtObCosAngle) {
                    isValid = true;
                    break;
                }
            }
            if (isValid){
                break;
            }
        }
        if (!isValid) {
            return false;
        }
    }
    return true;
}

template <typename Traits>
void ModelFilter<Traits>::makePtObs() {
    mPtObs.clear();
    mPtObs.resize(mSrc.pointList.size());
    for (IdxCap idxCap = 0; idxCap < mSrc.capList.size(); idxCap++){
        const auto& cap = mSrc.capList[idxCap];
        for (IdxLocalOb idxOb = 0; idxOb < cap.obs.size(); idxOb++){
            if (mCapObsValidMask[idxCap][idxOb]) {
                const auto &ob = cap.obs[idxOb];
                mPtObs[ob.ptIdx].push_back(PtOb{idxCap, idxOb});
            }
        }
    }
}

template <typename Traits>
HostModel<Traits> ModelFilter<Traits>::getNewModel() const {
    HostModel<Traits> dst{};
    auto copyValid = [](size_t& nbVar, auto& dstList, auto& dstVarMask, const auto& srcList, const auto& srcVarMask, const auto& validMask){
        nbVar = 0u;
        const size_t dstSize = std::count(validMask.begin(), validMask.end(), true);
        dstList.reserve(dstSize);
        dstVarMask.reserve(dstSize);
        for (size_t i = 0; i < srcList.size(); i++){
            if (validMask[i]) {
                dstList.push_back(srcList[i]);
                dstVarMask.push_back(srcVarMask[i]);
                if (srcVarMask[i]) {
                    nbVar++;
                }
            }
        }
    };
    const auto& src = mSrc;
    copyValid(dst.nbVarPoints, dst.pointList, dst.pointVarMask, src.pointList, src.pointVarMask, mPtValidMask);
    copyValid(dst.nbVarIntri, dst.intriList, dst.intriVarMask, src.intriList, src.intriVarMask, mCamValidMask);
    copyValid(dst.nbVarCaps, dst.capList, dst.capVarMask, src.capList, src.capVarMask, mCapValidMask);
    const auto camMap = getCamMap();
    const auto capMap = getCapMap();
    const auto ptMap = getPtMap();
    // fix camera index, capture observations and point index
    for (IdxCap idxCapSrc = 0; idxCapSrc < src.capList.size(); idxCapSrc++){
        if (!mCapValidMask[idxCapSrc]) {
            continue;
        }
        const auto& srcCap = src.capList[idxCapSrc];
        const IdxCap idxCapDst = capMap.at(idxCapSrc);
        assert(idxCapDst != badIdx<IdxCap>());
        assert(src.capVarMask[idxCapSrc] == dst.capVarMask[idxCapDst]);
        auto& dstCap = dst.capList.at(idxCapDst);
        if constexpr(isGroupModel<Traits>()) {
            dstCap.capture.intrIdx = camMap.at(srcCap.capture.intrIdx);
            assert(dstCap.capture.intrIdx != badIdx<IdxCam>());
        }
        dstCap.obs.clear();
        const auto obsValidMask = mCapObsValidMask[idxCapSrc];
        for (IdxLocalOb idxOb = 0; idxOb < srcCap.obs.size(); idxOb++){
            if (obsValidMask[idxOb]){
                auto ob = srcCap.obs[idxOb];
                ob.ptIdx = ptMap[ob.ptIdx];
                assert(ob.ptIdx != badIdx<IdxPt>());
                dstCap.obs.push_back(ob);
            }
        }
        dstCap.obs.shrink_to_fit();
    }
    dst.nbTotalObs = std::accumulate(dst.capList.begin(), dst.capList.end(), 0ul, [](size_t acc, auto& cap){return acc + cap.obs.size();});
    // fix point index for softCtrlPts
    for (const auto& [idxPtSrc, ctrlLoc] : src.softCtrlPts) {
        const auto [iter, success] = dst.softCtrlPts.try_emplace(ptMap.at(idxPtSrc), ctrlLoc);
        require(success);
    }
    return dst;
}

template <typename Index>
static std::vector<Index> makeMapFromMask(
        const std::vector<bool>& mask,
        const std::vector<bool>& varMask = {}) // used for sanity-checking
{
    for (Index idx = 0; idx < varMask.size(); idx++){
        if (!varMask[idx]){
            assert(mask.at(idx));
        }
    }
    std::vector<Index> result(mask.size(), badIdx<Index>());
    Index newIdx = 0;
    for (Index idx = 0; idx < mask.size(); idx++){
        if (mask[idx]){
            result[idx] = newIdx++;
        }
    }
    return result;
}

template <typename Traits>
std::vector<IdxCam> ModelFilter<Traits>::getCamMap() const {
    return makeMapFromMask<IdxCam>(mCamValidMask, mSrc.intriVarMask);
}

template <typename Traits>
std::vector<IdxCap> ModelFilter<Traits>::getCapMap() const {
    return makeMapFromMask<IdxCap>(mCapValidMask, mSrc.capVarMask);
}

template <typename Traits>
std::vector<IdxPt> ModelFilter<Traits>::getPtMap() const {
    return makeMapFromMask<IdxPt>(mPtValidMask, mSrc.pointVarMask);
}

template <typename Traits>
void ModelFilter<Traits>::apply(bool verbose){
    // filter out invalid observations
    size_t nbInvalidObs = 0;
    for (IdxCap i = 0; i < mSrc.capList.size(); i++) {
        const auto &cap = mSrc.capList[i];
        auto& obsValidMask = mCapObsValidMask.at(i);
        for (IdxLocalOb j = 0; j < cap.obs.size(); j++){
            if (obsValidMask[j]){
                if (!isValidOb(i, j)){
                    obsValidMask[j] = false;
                    nbInvalidObs++;
                }
            }
        }
    }
    if (verbose) {
        std::cout << format("Detected %u invalid observations") % nbInvalidObs << std::endl;
    }
    struct{
        std::unordered_set<IdxCam> cam;
        std::unordered_set<IdxCap> cap;
        std::unordered_set<IdxPt> pt;
    }candidates; // candidates to be checked.
    auto fillCandidates = [](auto& dst, const std::vector<bool>& varMask){
        dst.reserve(std::count(varMask.begin(), varMask.end(), true));
        for (uint32_t idx = 0; idx < varMask.size(); idx++){
            if (varMask[idx]){
                dst.insert(idx);
            }
        }
    };
    fillCandidates(candidates.cam, mSrc.intriVarMask);
    fillCandidates(candidates.cap, mSrc.capVarMask);
    fillCandidates(candidates.pt, mSrc.pointVarMask);
    auto invalidatePt = [this, &candidates](IdxPt idxPt){
//        std::cout << format("invalidating point %u") % idxPt << std::endl;
        assert(mSrc.pointVarMask.at(idxPt));
        mPtValidMask[idxPt] = false;
        candidates.pt.erase(idxPt);
        for (const auto& ob : mPtObs[idxPt]){
            mCapObsValidMask[ob.idxCap][ob.idxOb] = false;
            if (mCapValidMask[ob.idxCap] && mSrc.capVarMask[ob.idxCap]){
                candidates.cap.insert(ob.idxCap);
                const IdxCam idxCam = mSrc.capList.at(ob.idxCap).capture.intrIdx;
                if (isGroupModel<Traits>() && mCamValidMask[idxCam] && mSrc.intriVarMask[idxCam]) {
                    candidates.cam.insert(idxCam);
                }
            }
        }
    };
    auto invalidateCap = [this, &candidates](IdxCap idxCap){
//        std::cout << format("invalidating capture %u") % idxCap << std::endl;
        assert(mSrc.capVarMask[idxCap]);
        mCapValidMask.at(idxCap) = false;
        candidates.cap.erase(idxCap);
        auto& capObsValidMask = mCapObsValidMask[idxCap];
        const auto& cap = mSrc.capList[idxCap];
        for (IdxLocalOb idxOb = 0; idxOb < capObsValidMask.size(); idxOb++){
            if (capObsValidMask[idxOb]){
                capObsValidMask[idxOb] = false;
                const IdxPt idxPt = cap.obs[idxOb].ptIdx;
                if (mPtValidMask[idxPt] && mSrc.pointVarMask[idxPt]) {
                    candidates.pt.insert(idxPt);
                }
            }
        }
        const IdxCam idxCam = cap.capture.intrIdx;
        if (isGroupModel<Traits>() && mCamValidMask[idxCam] && mSrc.intriVarMask[idxCam]) {
            candidates.cam.insert(idxCam);
        }
    };
    auto invalidateCam = [this, &candidates](IdxCam idxCam){
//        std::cout << format("invalidating camera %u") % idxCam << std::endl;
        assert(mSrc.intriVarMask[idxCam]);
        mCamValidMask.at(idxCam) = false;
        candidates.cam.erase(idxCam);
        for (const IdxCap idxCap : mCamCaps.at(idxCam)){
            if (mCapValidMask[idxCap]) {
                const auto& cap = mSrc.capList.at(idxCap);
                if (mSrc.capVarMask[idxCap]){
                    candidates.cap.insert(idxCap);
                }
                for (const auto& ob : cap.obs){
                    if (mSrc.pointVarMask[ob.ptIdx]){
                        candidates.pt.insert(ob.ptIdx);
                    }
                }
            }
        }
    };
    while (!candidates.cam.empty() || !candidates.cap.empty() || !candidates.pt.empty())
    {
        auto checkAndInvalidate = [](auto& cand, auto& validMask, const auto& varMask, const auto& checkFunc, const auto& invalidateFunc){
            using Index = typename std::decay_t<decltype(cand)>::value_type;
            std::vector<Index> candCopy(cand.begin(), cand.end());
            std::sort(candCopy.begin(), candCopy.end());
            for (const auto idx : candCopy){
                assert(varMask[idx]);
                assert(validMask[idx]);
                if (checkFunc(idx)){
                    cand.erase(idx);
                }
                else{
                    invalidateFunc(idx);
                }
            }
        };
        // filter out invalid points
        checkAndInvalidate(candidates.pt, mPtValidMask, mSrc.pointVarMask, [this](IdxPt idxPt){return isValidPt(idxPt);}, invalidatePt);
        // filter out invalid captures
        checkAndInvalidate(candidates.cap, mCapValidMask, mSrc.capVarMask, [this](IdxCap idxCap){return isValidCap(idxCap);}, invalidateCap);
        // filter out invalid cameras
        checkAndInvalidate(candidates.cam, mCamValidMask, mSrc.intriVarMask, [this](IdxCam idxCam){return isValidCam(idxCam);}, invalidateCam);
    }
}

#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template class ModelFilter<TRAITS>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

} // namespace rba::grouped
