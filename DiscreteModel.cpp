//
// Created by yao on 9/15/19.
//
#include "DiscreteModel.h"
#include "GroupModel.h"
#include <boost/preprocessor/seq/for_each.hpp>
namespace rba
{
template<typename CaptureParamsType>
IModel::IntriType IDiscreteModel<CaptureParamsType>::getIntriType() const {
    return decltype(CaptureParamsType{}.intrinsics)::intriType;
}

template<typename Traits>
IdxPt DiscreteModel<Traits>::addPoint(double3 position, bool fixed) {
    return mGrpModel->addPoint(position, fixed);
}

template<typename Traits>
double3 DiscreteModel<Traits>::getPointPosition(IdxPt idx) const {
    return mGrpModel->getPointPosition(idx);
}

template<typename Traits>
void DiscreteModel<Traits>::setPointFixed(IdxPt idx, bool fixed) {
    mGrpModel->setPointFixed(idx, fixed);
}

template<typename Traits>
void DiscreteModel<Traits>::setCaptureFixed(IdxCap idx, bool fixed) {
    mGrpModel->setCaptureFixed(idx, fixed);
}

template <typename Traits>
void DiscreteModel<Traits>::setCaptureGNSS(IdxCap idx, double3 position, float omega[3][3], float huber) {
    mGrpModel->setCaptureGNSS(idx, position, omega, huber);
}

template <typename Traits>
void DiscreteModel<Traits>::setSoftCtrlPoint(IdxPt idx, double3 position, float omega[3][3], float huber) {
    mGrpModel->setSoftCtrlPoint(idx, position, omega, huber);
}

template<typename Traits>
void DiscreteModel<Traits>::initializeOptimization() {
    mGrpModel->initializeOptimization();
}

template<typename Traits>
void DiscreteModel<Traits>::setInitDamp(float damp) {
    mGrpModel->setInitDamp(damp);
}

template<typename Traits>
void DiscreteModel<Traits>::optimize(size_t maxIters) {
    mGrpModel->optimize(maxIters);
}

template<typename Traits>
IdxCap DiscreteModel<Traits>::addCapture(const CapParamType &params, bool fixed) {
    return mGrpModel->addCapture(badIdx<IdxCam>(), params, fixed);
}

template<typename Traits>
void DiscreteModel<Traits>::setCaptureParams(IdxCap idx, const CapParamType& params) {
    mGrpModel->setCaptureParams(idx, params);
}

template<typename Traits>
typename DiscreteModel<Traits>::CapParamType DiscreteModel<Traits>::getCaptureParams(IdxCap idx) const {
    return mGrpModel->getCaptureParams(idx);
}

template<typename Traits>
void DiscreteModel<Traits>::addObservation(IdxCap idxCap, IdxPt idxPt, float2 proj, float omega, float huber) {
    mGrpModel->addObservation(idxCap, idxPt, proj, omega, huber);
}

template<typename Traits>
DiscreteModel<Traits>::DiscreteModel()
: mGrpModel{std::make_unique<grouped::GroupModel<Traits>>()}{}

template<typename Traits>
void DiscreteModel<Traits>::filterModel() {
    mGrpModel->filterModel();
}

template<typename Traits>
DiscreteModel<Traits>::~DiscreteModel() = default;

template<typename Traits>
void DiscreteModel<Traits>::clear() {
    mGrpModel->clear();
}

template<typename Traits>
void DiscreteModel<Traits>::setVerbose(bool verbose) {
    mGrpModel->setVerbose(verbose);
}

#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template class DiscreteModel<TRAITS>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_DISCRETE_TRAITS)
#undef INSTANTIATE_TEMPLATES

}