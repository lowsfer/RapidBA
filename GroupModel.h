//
// Created by yao on 27/11/18.
//

#pragma once
#include "RapidBA.h"
#include <vector>
#include "utils_host.h"
#include "containers.h"
#include "computeHessian.h"
#include "fwd.h"
#include <optional>

namespace rba {
namespace grouped {

template <typename Traits>
struct HostModel {
    RBA_IMPORT_TRAITS(Traits);
    std::vector<rba::Point<lpf>> pointList;
    std::vector<bool> pointVarMask;
    size_t nbVarPoints = 0;
    // key is index to pointList, which is different from DeviceModel
    std::unordered_map<uint32_t, rba::CtrlLoc<lpf>> softCtrlPts;

    std::vector<CamIntr> intriList;
    std::vector<bool> intriVarMask;
    size_t nbVarIntri = 0;

    struct CaptureViews {
        typename Traits::Capture capture;
        std::vector<rba::CapOb<lpf>> obs;//should be sorted by order in pointList, i.e. ptIdx
        std::optional<rba::CtrlLoc<lpf>> gnss;
    };
    rba::IdxOb nbTotalObs = 0;
    std::vector<CaptureViews> capList;
    std::vector<bool> capVarMask;
    size_t nbVarCaps = 0;

    void clearData() {
        pointList.clear();
        pointVarMask.clear();
        nbVarPoints = 0;
		softCtrlPts.clear();
        intriList.clear();
        intriVarMask.clear();
        nbVarIntri = 0;
        nbTotalObs = 0;
        capList.clear();
        capVarMask.clear();
        nbVarCaps = 0;
    }
};

template <typename Traits>
class GroupModel : public Traits::Interface, protected HostModel<Traits> {
public:
    RBA_IMPORT_FPTYPES(Traits);
    using Interface = typename Traits::Interface;
    using typename Interface::CamParamType;
    using typename Interface::CapParamType;

    GroupModel();
    ~GroupModel();

    IdxCam addCamera(const CamParamType& params, bool fixed) override;

    IdxPt addPoint(double3 position, bool fixed) override;

    IdxCap addCapture(IdxCam idxCam, const CapParamType &params, bool fixed) override;

    void addObservation(IdxCap idxCap, IdxPt idxPt, float2 proj, float omega, float huber) override;

    void setCameraFixed(IdxCam idx, bool fixed) override;

    void setCaptureFixed(IdxCap idx, bool fixed) override;

    void setPointFixed(IdxPt idx, bool fixed) override;

    // Use omega = INFINITY if you want the capture location to be fixed
    void setCaptureGNSS(IdxCap idx, double3 position, float omega[3][3], float huber) override;
    // omega must be positive finite. For hard control point, use setPointFixed() instead.
    void setSoftCtrlPoint(IdxPt idx, double3 position, float omega[3][3], float huber) override;

    IdxCam getIdxCamForCapture(IdxCap idx) const override;

    CamParamType getCameraParams(IdxCam idx) const override;

    void setCaptureParams(IdxCap idx, const CapParamType& params) override;

    CapParamType getCaptureParams(IdxCap idx) const override;

    double3 getPointPosition(IdxPt idx) const override;

    void setInitDamp(float damp) override;

    void initializeOptimization() override;

    void optimize(size_t maxIters/* = 64*/) override;

    void filterModel() override;

    void clear() override;

    void setVerbose(bool verbose) override;

    [[nodiscard]] typename IModel::IntriType getIntriType() const override;
protected:
    void sortObservations();
    static Coordinate<lpf> toCoord(double x) {
        return {static_cast<typename Coordinate<lpf>::ImplType>(x)};
    }

protected:
    bool mVerbose = false;
    // Mapping from API-returned index to the new filtered index
    std::vector<IdxCam> mCamMap;
    std::vector<IdxCap> mCapMap;
    std::vector<IdxPt> mPtMap;
    // Inverse of mCamMap;
    std::vector<IdxCam> mInvCamMap;
    std::unique_ptr<HostModel<Traits>> mOriginalModel;// backup for the original model before filter

    cudaStream_t stream = nullptr;
    std::unique_ptr<BundleSolver<Traits>> solver;

    using HostModel<Traits>::nbVarPoints;
    using HostModel<Traits>::nbVarCaps;
    using HostModel<Traits>::nbVarIntri;
    using HostModel<Traits>::nbTotalObs;
    using HostModel<Traits>::intriList;
    using HostModel<Traits>::intriVarMask;
    using HostModel<Traits>::capList;
    using typename HostModel<Traits>::CaptureViews;
    using HostModel<Traits>::capVarMask;
    using HostModel<Traits>::pointList;
    using HostModel<Traits>::pointVarMask;

    void syncStream() const{
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
};

} // namespace grouped
} //namespace rba