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
// Discrete models are actually implemented with GroupModel

#pragma once
#include <vector>
#include "RapidBA.h"
#include "fwd.h"
#include <memory>

namespace rba
{

namespace grouped{
template <bool rolling = false>
struct CamParamTypeNull{
    static constexpr IModel::IntriType intriType = IModel::IntriType::kNull;
	std::monostate rollingCenter;
};
} // namespace grouped

using IDiscreteModelF1D2Impl = IGroupModel<grouped::CamParamTypeNull, discrete::CapParamTypeF1D2, ShutterType::kGlobal>;
using IDiscreteModelF1Impl = IGroupModel<grouped::CamParamTypeNull, discrete::CapParamTypeF1, ShutterType::kGlobal>;
using IDiscreteModel0Impl = IGroupModel<grouped::CamParamTypeNull, discrete::CapParamType0, ShutterType::kGlobal>;

template <typename Traits>
class DiscreteModel : public Traits::PubInterface
{
public:
    using typename Traits::PubInterface::CapParamType;
    DiscreteModel();
    ~DiscreteModel() override;

    IdxPt addPoint(double3 position, bool fixed) override;
    double3 getPointPosition(IdxPt idx) const override;

    void setPointFixed(IdxPt idx, bool fixed) override;
    void setCaptureFixed(IdxCap idx, bool fixed) override;

    // Use omega = INFINITY if you want the capture location to be fixed
    void setCaptureGNSS(IdxCap idx, double3 position, float omega[3][3], float huber) override;
    // omega must be positive finite. For hard control point, use setPointFixed() instead.
    void setSoftCtrlPoint(IdxPt idx, double3 position, float omega[3][3], float huber) override;

    void filterModel() override;
    void initializeOptimization() override;
    void setInitDamp(float damp) override; // Default: 1E-3f. Must init solver first
    void optimize(size_t maxIters/* = 64*/) override;
    IdxCap addCapture(const CapParamType& params, bool fixed) override;
    void setCaptureParams(IdxCap idx, const CapParamType& params) override ;
    CapParamType getCaptureParams(IdxCap idx) const override;
    void addObservation(IdxCap idxCap, IdxPt idxPt, float2 proj, float omega/* = 1.f*/, float huber/* = INFINITY*/) override;
    void clear() override;
    void setVerbose(bool verbose) override;
private:
    std::unique_ptr<grouped::GroupModel<Traits>> mGrpModel;
};


} // namespace rba
