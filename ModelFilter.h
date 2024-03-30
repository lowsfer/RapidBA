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
// Created by yao on 9/21/19.
//

#pragma once
#include "GroupModel.h"
#include <vector>

namespace rba::grouped {

template <typename Traits>
class ModelFilter
{
public:
    explicit ModelFilter(const HostModel<Traits>& src)
            : mSrc{src}
            , mCapValidMask(src.capList.size(), true)
            , mPtValidMask(src.pointList.size(), true)
            , mCamValidMask(src.intriList.size(), true)
    {
        mCapObsValidMask.reserve(src.capList.size());
        mCamCaps.resize(src.intriList.size());
        for (size_t i = 0; i < src.capList.size(); i++){
            const auto& cap = src.capList[i];
            mCapObsValidMask.emplace_back(cap.obs.size(), true);
            if constexpr (isGroupModel<Traits>()){
                mCamCaps.at(cap.capture.intrIdx).push_back(i);
            }
        }
        makePtObs();
    }
    using IdxLocalOb = uint32_t;
    struct PtOb{
        IdxCap idxCap;
        IdxLocalOb idxOb;
    };

    void apply(bool verbose = false);
    [[nodiscard]] HostModel<Traits> getNewModel() const;
    [[nodiscard]] std::vector<IdxCam> getCamMap() const;
    [[nodiscard]] std::vector<IdxCap> getCapMap() const;
    [[nodiscard]] std::vector<IdxPt> getPtMap() const;

private:
    // these functions do not consider existing mCapValidMask/mPtValidMask/mCamValidMask
    [[nodiscard]] bool isValidOb(IdxCap idxCap, IdxLocalOb idxOb) const;
    [[nodiscard]] bool isValidCap(IdxCap idxCap) const;
    [[nodiscard]] bool isValidPt(IdxPt idxPt) const;
    [[nodiscard]] bool isValidCam(IdxCam idxCam) const;
    void makePtObs();

    const HostModel<Traits>& mSrc;
    std::vector<bool> mCapValidMask;
    std::vector<bool> mPtValidMask;
    std::vector<bool> mCamValidMask;
    std::vector<std::vector<bool>> mCapObsValidMask;
    std::vector<std::vector<PtOb>> mPtObs;
    std::vector<std::vector<IdxCap>> mCamCaps;
    float mRefDistance  = 1.f; // typical distance between two neighbour captures
    float mMinCapPtRelativeDistance = 1E-2f;
    size_t mMinPtObs = 2u;
    float mMaxPtObCosAngle = std::cos(0.1f/180*float(M_PI)); // min cap-point-cap angle requires for a point to be valid
    size_t mMinCapObs = divUp(Traits::Capture::DoF + Traits::CamIntr::DoF + 1, 2u); // +1 to remove P3P ambiguity
    float mMinCapObObSinAngle = std::sin(0.1f / 180 * float(M_PI));
    float mMaxCapObTanAngle = std::tan(75.f / 180 * float(M_PI));
};
}