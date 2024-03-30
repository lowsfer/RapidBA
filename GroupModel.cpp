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
// Created by yao on 27/11/18.
//

#include "GroupModel.h"
#include "GroupModelTypes.h"
#include "BundleSolver.h"
#include "blockSolvers/HessianSolver.h"
#include <boost/preprocessor/seq/for_each.hpp>
#include "traits/pubParamConvert.h"
#include "ModelFilter.h"
#include <iostream>
#include <boost/format.hpp>
#include <eigen3/Eigen/Dense>
using format = boost::format;

namespace rba::grouped{

half checkedFloat2Half(float x)
{
    const half y = half(x);
    require(std::isfinite(float(y)) == std::isfinite(x));
    return y;
}

template <typename ftype>
CtrlLoc<ftype> makeCtrlLoc(const double3& loc, float omega[3][3], float huber) {
    using locf = Coordinate<ftype>;
    using LocfImpl = typename locf::ImplType;
    return {
        {
            locf{static_cast<LocfImpl>(loc.x)},
            locf{static_cast<LocfImpl>(loc.y)},
            locf{static_cast<LocfImpl>(loc.z)}
        },
        huber,
        symkmat<ftype, 3>{kmat<float, 3, 3>{omega}.cast<ftype>()}
    };
}

template <typename Traits>
GroupModel<Traits>::GroupModel() {
    nbVarPoints = 0;
    nbVarCaps = 0;
    nbVarIntri = 0;
    nbTotalObs = 0;
}

template <typename Traits>
GroupModel<Traits>::~GroupModel() = default;

template <typename Traits>
IdxCam GroupModel<Traits>::addCamera(const CamParamType& params, bool fixed) {
    const auto idx = numeric_cast<IdxCam>(intriList.size());
    intriList.emplace_back(Traits::toCamParams(params));
    intriVarMask.push_back(!fixed);
    if (!fixed)
        nbVarIntri++;
    mCamMap.push_back(idx);
    mInvCamMap.push_back(idx);
    return idx;
}

template <typename Traits>
IdxCap GroupModel<Traits>::addCapture(IdxCam idxCam, const CapParamType &params, bool fixed) {
    const auto idx = numeric_cast<IdxCap>(capList.size());
    capList.emplace_back(CaptureViews{makeCapParams<typename Traits::Capture>(params, idxCam), {}});
    capVarMask.push_back(!fixed);
    if (!fixed)
        nbVarCaps++;
    mCapMap.push_back(idx);
    return idx;
}

template <typename Traits>
IdxPt GroupModel<Traits>::addPoint(double3 position, bool fixed) {
    const auto idx = numeric_cast<IdxPt>(pointList.size());
    pointList.emplace_back(rba::Point<typename Traits::lpf>{toCoord(position.x), toCoord(position.y), toCoord(position.z)});
    pointVarMask.push_back(!fixed);
    if (!fixed)
        nbVarPoints++;
    mPtMap.push_back(idx);
    return idx;
}

template <typename Traits>
void GroupModel<Traits>::addObservation(rba::IdxCap idxCap, rba::IdxPt idxPt, float2 proj, float omega, float huber) {
    const half omegaF16 = checkedFloat2Half(omega);
	assert(std::isfinite(float(omegaF16)));
    const half huberF16 = checkedFloat2Half(huber);
    capList[idxCap].obs.emplace_back(rba::CapOb<decltype(proj.x)>{{proj.x, proj.y}, idxPt, omegaF16, huberF16});
    nbTotalObs++;
}

template <typename Traits>
void GroupModel<Traits>::setCameraFixed(rba::IdxCam idx, bool fixed) {
    if (fixed) {
        if (intriVarMask[idx])
            nbVarIntri--;
    }
    else{
        if (!intriVarMask[idx])
            nbVarIntri++;
    }
    intriVarMask[idx] = !fixed;
}

template <typename Traits>
void GroupModel<Traits>::setCaptureFixed(rba::IdxCap idx, bool fixed) {
    if (fixed) {
        if (capVarMask[idx])
            nbVarCaps--;
    }
    else{
        if (!capVarMask[idx])
            nbVarCaps++;
    }
    capVarMask[idx] = !fixed;
}

template <typename Traits>
void GroupModel<Traits>::setPointFixed(rba::IdxPt idx, bool fixed) {
    if (fixed) {
        if (pointVarMask[idx])
            nbVarPoints--;
    }
    else{
        if (!pointVarMask[idx])
            nbVarPoints++;
    }
    pointVarMask[idx] = !fixed;
}

template <typename Traits>
void GroupModel<Traits>::setCaptureGNSS(IdxCap idx, double3 position, float omega[3][3], float huber) {
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> o(&omega[0][0]);
    require(o.allFinite() && o.isApprox(o.transpose()) && o.llt().info() == Eigen::Success);
    auto& cap = capList.at(idx);
    for (int i = 0; i < 3; i++) {
        if (omega[i][i] > std::numeric_limits<float>::max()) {
            require(toCoord((&position.x)[i]) == cap.capture.pose.c[i]);
        }
    }
    require(cap.gnss == std::nullopt);
    cap.gnss = makeCtrlLoc<lpf>(position, omega, huber);
}

template <typename Traits>
void GroupModel<Traits>::setSoftCtrlPoint(IdxPt idx, double3 position, float omega[3][3], float huber) {
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> o(&omega[0][0]);
    require(o.allFinite() && o.isApprox(o.transpose()) && o.llt().info() == Eigen::Success);
    const auto [iter, success] = this->softCtrlPts.try_emplace(idx, makeCtrlLoc<lpf>(position, omega, huber));
    require(success);
}

template <typename Traits>
typename GroupModel<Traits>::CamParamType GroupModel<Traits>::getCameraParams(IdxCam idx) const {
    const IdxCap idxFiltered = mCamMap.at(idx);
    if (idxFiltered == badIdx<IdxCam>()){
        return Traits::toCamParams(mOriginalModel->intriList[idx]);
    }
    const auto& intri = intriList[idxFiltered];
    return Traits::toCamParams(intri);
}

template <typename Traits>
typename GroupModel<Traits>::CapParamType GroupModel<Traits>::getCaptureParams(rba::IdxCap idx) const {
    const IdxCap idxFiltered = mCapMap.at(idx);
    if (idxFiltered == badIdx<IdxCap>()){
        return convert(mOriginalModel->capList[idx].capture);
    }
    const auto cap = capList[idxFiltered].capture;
    return convert(cap);
}

template <typename Traits>
double3 GroupModel<Traits>::getPointPosition(rba::IdxPt idx) const {
    const IdxCap idxFiltered = mPtMap.at(idx);
    const auto& pt = (idxFiltered == badIdx<IdxPt>()) ? mOriginalModel->pointList[idx] : pointList[idxFiltered];
    return {pt.position[0].value, pt.position[1].value, pt.position[2].value};
}

template <typename Traits>
void GroupModel<Traits>::sortObservations() {
    for (auto& cap : capList)
        std::sort(cap.obs.begin(), cap.obs.end(), [](const rba::CapOb<lpf>& a, const rba::CapOb<lpf>& b)->bool{return a.ptIdx < b.ptIdx;});
}

template <typename Traits>
void GroupModel<Traits>::initializeOptimization() {
    sortObservations();

    if (solver == nullptr){
        solver = std::make_unique<BundleSolver<Traits>>();
        solver->setVerbose(mVerbose);
    }
    solver->initialize(this);
    assert(nbVarPoints == numeric_cast<size_t>(std::count(pointVarMask.begin(), pointVarMask.end(), true)));
}

template <typename Traits>
void GroupModel<Traits>::optimize(size_t maxIters) {
    if (nbVarIntri == 0 && nbVarCaps == 0 && nbVarPoints == 0){
        return;
    }
    solver->solve(maxIters, stream);
    syncStream();
}

template <typename Traits>
void GroupModel<Traits>::clear() {
    mCamMap.clear();
    mCapMap.clear();
    mPtMap.clear();
    mInvCamMap.clear();
    mOriginalModel.reset();
    HostModel<Traits>::clearData();
    if (solver != nullptr) solver->clear();
}

template <typename Traits>
void GroupModel<Traits>::setVerbose(bool verbose) {
    mVerbose = verbose;
    if (solver != nullptr) {
        solver->setVerbose(verbose);
    }
}

template <typename Traits>
void GroupModel<Traits>::setInitDamp(float damp) {
    solver->setInitDamp(damp);
}

template <typename Traits>
void GroupModel<Traits>::filterModel() {
    sortObservations();
    ModelFilter<Traits> filter(*this);
    filter.apply();
    mCamMap = filter.getCamMap();
    mCapMap = filter.getCapMap();
    mPtMap = filter.getPtMap();
    mInvCamMap.clear();
    for (IdxCam apiIdx = 0; apiIdx < mCamMap.size(); apiIdx++) {
        const IdxCam filteredIdx = mCamMap.at(apiIdx);
        if (filteredIdx != badIdx<IdxCam>()) {
            if (mInvCamMap.size() < filteredIdx + 1){
                mInvCamMap.insert(mInvCamMap.end(), filteredIdx + 1 - mInvCamMap.size(), badIdx<IdxCam>());
            }
            mInvCamMap.at(filteredIdx) = apiIdx;
        }
    }
    require(std::all_of(mInvCamMap.begin(), mInvCamMap.end(), [this](IdxCam i){
        return i >= 0 && i < mCamMap.size();
    }));
    auto newModel = filter.getNewModel();
    mOriginalModel.reset(new HostModel<Traits>(std::move(*this)));
    static_cast<HostModel<Traits>&>(*this) = std::move(newModel);
    if (mVerbose) {
        if constexpr(isGroupModel<Traits>())
            std::cout << format("New model has %u cameras, %u captures, %u points and %u observations") %
                         intriList.size() % capList.size() % pointList.size() % nbTotalObs << std::endl;
        else
            std::cout << format("New model has %u captures, %u points and %u observations") % capList.size() %
                         pointList.size() % nbTotalObs << std::endl;
    }
}

template <typename Traits>
typename IModel::IntriType GroupModel<Traits>::getIntriType() const {
    return Interface::CamParamType::intriType;
}

template<typename Traits>
void GroupModel<Traits>::setCaptureParams(IdxCap idxCap, const CapParamType &params) {
    typename Traits::Capture* cap = nullptr;
    if (mOriginalModel != nullptr) {
        auto& origCap = mOriginalModel.get()->capList.at(idxCap).capture;
        const uint32_t idxCam = origCap.intrIdx;
        origCap = makeCapParams<typename Traits::Capture>(params, idxCam);
        if (mCapMap.at(idxCap) != badIdx<IdxCap>()) {
            cap = &capList.at(mCapMap.at(idxCap)).capture;
        }
    }
    else {
        cap = &capList.at(idxCap).capture;
    }
    if (cap != nullptr)
    {
        const uint32_t idxCam = cap->intrIdx;
        *cap = makeCapParams<typename Traits::Capture>(params, idxCam);
    }
}

template<typename Traits>
IdxCam GroupModel<Traits>::getIdxCamForCapture(IdxCap idx) const {
    const IdxCap idxFiltered = mCapMap.at(idx);
    const auto& cap = (idxFiltered == badIdx<IdxPt>()) ? mOriginalModel->capList[idx] : capList[idxFiltered];
    const IdxCam idxIntriFiltered = cap.capture.intrIdx;
    const IdxCam idxCam = mInvCamMap.at(idxIntriFiltered);
    return idxCam;
}

#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template class GroupModel<TRAITS>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

}//rba::grouped
