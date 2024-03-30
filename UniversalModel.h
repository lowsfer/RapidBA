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
// Created by yao on 2/23/20.
//

#pragma once
#include "RapidBA.h"
#include <memory>
#include <unordered_map>

namespace rba {
class UniversalModel : public IUniversalModel {
public:
    using Interface = IUniversalModel;
    using typename Interface::CamParamType;
    using typename Interface::CapParamType;

    UniversalModel(bool useGroupModel, IntriType intriType, ShutterType shutter);
    ~UniversalModel() override;

    IdxCam addCamera(const CamParamType& params, bool fixed) override;

    IdxPt addPoint(double3 position, bool fixed) override;

    IdxCap addCapture(IdxCam idxCam, const CapParamType &params, bool fixed) override;

    void addObservation(IdxCap idxCap, IdxPt idxPt, float2 proj, float omega, float huber) override;

    void setCameraFixed(IdxCam idx, bool fixed) override;

    void setCaptureParams(IdxCap idx, const CapParamType& params) override;

    void setCaptureFixed(IdxCap idx, bool fixed) override;

    void setPointFixed(IdxPt idx, bool fixed) override;

    // Use omega = INFINITY if you want the capture location to be fixed
    void setCaptureGNSS(IdxCap idx, double3 position, float omega[3][3], float huber) override;
    // omega must be positive finite. For hard control point, use setPointFixed() instead.
    void setSoftCtrlPoint(IdxPt idx, double3 position, float omega[3][3], float huber) override;

    IdxCam getIdxCamForCapture(IdxCam idx) const override;

    CamParamType getCameraParams(IdxCam idx) const override;

    CapParamType getCaptureParams(IdxCap idx) const override;

    double3 getPointPosition(IdxPt idx) const override;

    void setInitDamp(float damp) override;

    void initializeOptimization() override;

    void optimize(size_t maxIters/* = 64*/) override;

    void filterModel() override;

    void clear() override;

    void setVerbose(bool verbose) override;

    [[nodiscard]] IntriType getIntriType() const final;

    [[nodiscard]] bool isGrouped() const override;

	ShutterType getShutterType() const final {return mModel->getShutterType();}

private:
	template <ShutterType shutter> IdxCam addCameraImpl(const CamParamType& params, bool fixed);
	template <ShutterType shutter> CamParamType getCameraParamsImpl(IdxCam idx) const;
	template <ShutterType shutter> IdxCap addCaptureImpl(IdxCam idxCam, const CapParamType &params, bool fixed);
	template <ShutterType shutter> CapParamType getCaptureParamsImpl(IdxCap idx) const;
	template <ShutterType shutter> void setCaptureParamsImpl(IdxCap idxCap, const IModel::Pose<true> &params);
private:
    bool mIsGrouped;
    IntriType mIntriType{IntriType::kNull};
    std::unique_ptr<IMonoModel> mModel;
    std::unordered_map<IdxCam, float2> mCamCenters; // key is in original IdxCam, before model filter.
};

} // namespace rba
