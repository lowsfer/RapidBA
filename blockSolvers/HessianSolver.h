//
// Created by yao on 6/12/18.
//

#pragma once
#include <memory>
#include <cuda_runtime_api.h>
#include "solveHessian.h"
#include "solveSchurPCG.h"
#include "../containers.h"
#include "../computeHessian.h"
#include "../fwd.h"

namespace rba {
namespace solver {
using grouped::HostModel;
using grouped::DeviceModel;
using grouped::DeviceHessianPreCompData;
using grouped::DeviceHessian;
using grouped::DevHessianVec;

template <typename Traits>
class HessianSolver
{
public:
    enum class Type{
        //implicitSchur, //not yet implemented
        explicitSchur
    };
    virtual void clear() = 0;
    virtual void setUp(const rba::grouped::DeviceHessian<Traits> *hessian, const HessianVec<Traits, false> &x,
                       const rba::grouped::DeviceModel<Traits> *model, cudaStream_t stream) = 0;
    struct Threshold {typename HessianBase<Traits>::EcBlock c; typename HessianBase<Traits>::EaBlock a; typename HessianBase<Traits>::EbBlock b;};// b is not used for Schur-based solvers
    virtual bool solve(const Threshold &threshold, uint32_t maxIters, cudaStream_t stream) = 0;
    virtual ~HessianSolver() = default;
};

template <typename Traits>
std::unique_ptr<HessianSolver<Traits>> createHessianSolver(typename HessianSolver<Traits>::Type type = HessianSolver<Traits>::Type::explicitSchur);

template <typename Traits>
class HessianSolverWSchur : public HessianSolver<Traits>
{
    RBA_IMPORT_FPTYPES(Traits);
public:
    void clear() override;
    void setUp(const rba::grouped::DeviceHessian<Traits> *hessian, const HessianVec<Traits, false> &x,
               const rba::grouped::DeviceModel<Traits> *model, cudaStream_t stream) override;
protected:
    void computeInvV(cudaStream_t stream) const;
    // find solution for points, i.e. hessianX.b
    void solveB(cudaStream_t stream);

    cudaError_t migrateToDevice(int deviceId, cudaStream_t stream) const{
        checkCudaErrors(sparseIdxW2idxLocalOb->migrateToDevice(deviceId, stream));
        return cudaSuccess;
    }
protected:
    const DeviceHessian<Traits>* devHessian = nullptr;
    HessianVec<Traits, false> hessianX = {};//solution
    const DeviceModel<Traits>* devModel = nullptr;
    using VSymBlock = symkmat<lpf, rba::Point<lpf>::DoF>;
    std::unique_ptr<VSymBlock[], CudaDeviceDeleter> devInvVBlocks;
    size_t devInvVBlocksCapacity {0};
    std::shared_ptr<DevVecVec<uint16_t>> sparseIdxW2idxLocalOb;
};

}//namespace solver
}//namespace rba

