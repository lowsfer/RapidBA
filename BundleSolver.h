//
// Created by yao on 8/12/18.
//

#pragma once
#include <cuda_runtime_api.h>
#include <memory>
#include "utils_host.h"
#include "fwd.h"

namespace rba {
namespace grouped {

template <typename Traits>
class BundleSolver {
public:
    enum class DampType{
        diag,
        blockDiag // @todo: implement this and try
    };

    void clear();

    void initialize(HostModel<Traits> *model);

    void setInitDamp(float damp) {tao = damp;}

    void solve(uint32_t maxIters = 60, cudaStream_t stream = nullptr);

    void setVerbose(bool verbose) {mVerbose = verbose;}

    virtual ~BundleSolver();

protected:
    bool checkCondition1(cudaStream_t stream) const;
    bool checkCondition2(cudaStream_t stream) const; //Need to save model variables to devModelVarBackup before calling this
    bool checkCondition3(cudaStream_t stream) const;
    bool checkCondition4(cudaStream_t stream) const;

    void computeHessian(bool computeErrSqrNorm, cudaStream_t stream) const;
    void computeErrSqrNorm(cudaStream_t stream) const;
    void dampHessian(bool fromBackup, float damp, cudaStream_t stream) const;

protected:
    bool mVerbose {false};
    HostModel<Traits> *hostModel = nullptr;
    std::unique_ptr<DeviceModel<Traits>> devModel;
    std::unique_ptr<DeviceHessian<Traits>> devHessian;
    std::unique_ptr<DeviceHessianPreCompData> devHessianPreComp;
    std::unique_ptr<DevHessianVec<Traits>> devHessianDiagBackup;
    std::unique_ptr<DevHessianVec<Traits>> delta; //solution
    std::unique_ptr<DeviceModelVarBackup<Traits>> devModelVarBackup;
    std::unique_ptr<double, CudaDeviceDeleter> devErrSqrNorm = deviceAlloc<double>(1);
    std::unique_ptr<double, CudaHostDeleter> hostErrSqrNormLast = hostAlloc<double>(1);
    std::unique_ptr<double, CudaHostDeleter> hostErrSqrNormNew = hostAlloc<double>(1);

    float tao = 1E-3f;
    float threshold1 = 1E-6f;
    float threshold2 = 1E-6f;
    float threshold3 = 1E-6f;
    float threshold4 = 0.f;
    uint32_t maxIterPCG = 20u;
    float maxDamp = 1E4f;
    std::unique_ptr<float[], CudaDeviceDeleter> devScalar {deviceAlloc<float>(2).release()};
    std::unique_ptr<float[], CudaHostDeleter> hostScalar {hostAlloc<float>(2).release()};
    std::unique_ptr<double[], CudaHostDeleter> hostDoubleScalar {hostAlloc<double>(2).release()};

    std::unique_ptr<solver::HessianSolver<Traits>> hessianSolver;

    void syncStream(cudaStream_t stream) const{
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
};
}//namespace grouped
}//namespace rba

