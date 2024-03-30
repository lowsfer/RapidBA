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
// Created by yao on 8/12/18.
//

#include "BundleSolver.h"
#include "blockSolvers/HessianSolver.h"
#include "GroupModel.h"
#include "GroupModelTypes.h"
#include "blockSolvers/HessianSolver.h"
#include "computeHessian.h"
#include <stdexcept>
#include <iostream>
#include <boost/format.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
using boost::format;

#ifdef NDEBUG
static constexpr bool computeHessianW = false;
#else
static constexpr bool computeHessianW = true;
#endif

namespace rba{
template <typename Traits>
cudaError_t launchCudaGetAbsMax(float* absMaxVal, const HessianVec<Traits, true>& vec, cudaStream_t stream);
template <typename Traits>
cudaError_t launchCudaGetSquaredNorm(float* squaredNorm, const HessianVec<Traits, true>& vec, cudaStream_t stream);
template <typename Traits>
cudaError_t launchCudaGetModelVarSquaredNorm(float* varSquaredNorm, const updateModel::BackupData<Traits, true>& backup, cudaStream_t stream);
namespace grouped{

template <typename Traits>
BundleSolver<Traits>::~BundleSolver() = default;

template <typename Traits>
void BundleSolver<Traits>::clear(){
    hostModel = nullptr;
    if (devModel != nullptr) devModel->clear();
    if (devHessian != nullptr) devHessian->clear();
    if (devHessianPreComp != nullptr) devHessianPreComp->clear();
    if (devHessianDiagBackup != nullptr) devHessianDiagBackup->clear();
    if (delta != nullptr) delta->clear();
    if (devModelVarBackup != nullptr) devModelVarBackup->clear();
    if (hessianSolver != nullptr) hessianSolver->clear();
}

template <typename Traits>
void BundleSolver<Traits>::initialize(rba::solver::HostModel<Traits>* model_) {
    hostModel = model_;
    const rba::solver::HostModel<Traits>* model = model_;
    for (auto& cap : model->capList) {
        assert(std::is_sorted(cap.obs.begin(), cap.obs.end(), [](const rba::CapOb<typename Traits::lpf> &a,
                                                                 const rba::CapOb<typename Traits::lpf> &b) -> bool {
            return a.ptIdx < b.ptIdx;
        }));
        unused(cap);
    }

    if (devModel == nullptr)
        devModel = std::make_unique<DeviceModel<Traits>>();
    devModel->init(*model);

    if (devHessian == nullptr)
        devHessian = std::make_unique<DeviceHessian<Traits>>();
    if (devHessianPreComp == nullptr)
        devHessianPreComp = std::make_unique<DeviceHessianPreCompData>();
    devHessian->init(*model, devModel->involvedCaps.data(), devModel->involvedCaps.size(), devHessianPreComp.get(), computeHessianW);
    if (devHessianDiagBackup == nullptr)
        devHessianDiagBackup = std::make_unique<DevHessianVec<Traits>>();
    devHessianDiagBackup->resize(cast32u(model->nbVarIntri), cast32u(model->nbVarCaps), cast32u(model->nbVarPoints));

    assert(model->nbVarPoints == numeric_cast<size_t>(std::count(model->pointVarMask.begin(), model->pointVarMask.end(), true)));

    const auto solverType = solver::HessianSolver<Traits>::Type::explicitSchur;
    if (hessianSolver == nullptr)
        hessianSolver = solver::createHessianSolver<Traits>(solverType);
    if (delta == nullptr)
        delta = std::make_unique<DevHessianVec<Traits>>();
    delta->resize(cast32u(model->nbVarIntri), cast32u(model->nbVarCaps), cast32u(model->nbVarPoints));
    delta->migrateToDevice(getCudaDevice(), nullptr);
    hessianSolver->setUp(devHessian.get(), delta->getParam(), devModel.get(), nullptr);

    if (devModelVarBackup == nullptr){
        devModelVarBackup = std::make_unique<DeviceModelVarBackup<Traits>>();
    }
    devModelVarBackup->resize(*devModel);
}

template <typename Traits>
void BundleSolver<Traits>::computeHessian(bool computeErrSqrNorm, cudaStream_t stream) const {
    checkCudaErrors(rba::launchCudaComputeHessian(devModel->template getKernelArgs<true>(), computeErrSqrNorm ? devErrSqrNorm.get() : nullptr, devHessian->getKernelArgs(),
                                                  cast32u(devHessian->S.data.size()), devHessianPreComp->getKernelArgs(), stream));
}

template <typename Traits>
void BundleSolver<Traits>::computeErrSqrNorm(cudaStream_t stream) const {
    checkCudaErrors(rba::launchCudaComputeErrSqrNorm(devModel->template getKernelArgs<true>(), devErrSqrNorm.get(), stream));
}

//@fixme: move some code from GroupModel.cpp here
template <typename Traits>
void BundleSolver<Traits>::solve(uint32_t maxIters, cudaStream_t stream) {
    {
        const int device = getCudaDevice();
        checkCudaErrors(devModel->migrateToDevice(device, stream));
        checkCudaErrors(devHessian->migrateToDevice(device, stream));
        checkCudaErrors(devHessianPreComp->migrateToDevice(device, stream));
        checkCudaErrors(devHessianDiagBackup->migrateToDevice(device, stream));
        checkCudaErrors(delta->migrateToDevice(device, stream));
    }

    float v = 2.f;
    computeHessian(true, stream);
    checkCudaErrors(cudaMemcpyAsync(hostErrSqrNormNew.get(), devErrSqrNorm.get(), sizeof(*devErrSqrNorm), cudaMemcpyDeviceToHost, stream));
    syncStream(stream);
    double chi2 = *hostErrSqrNormNew;
    if (!std::isfinite(chi2)){
        std::cout << "The squared error is not finite. Cannot continue.(" << chi2 << ")"  << std::endl;
        return;
    }
    bool stop = checkCondition1(stream);

    // @fixme: this parameter may need tuning.
    float mu = tao * 1.f; //@info We use diag+mu*sqrt(diag), i.e. the PBA approach, rather than diag+mu*I (SBA).
    uint32_t nbAcceptedSteps = 0u;
    uint32_t nbRejectedSteps = 0u;
    std::string stopReason; // @fixme: change to enum
    for (uint32_t k = 0; k < maxIters; k++){
        if (mVerbose) {
            std::cout << format("LM step %u: chi2 = %e, lambda = %e") % k % chi2 % mu << std::endl;
        }
        checkCudaErrors(cudaMemcpyAsync(hostErrSqrNormLast.get(), devErrSqrNorm.get(), sizeof(*devErrSqrNorm), cudaMemcpyDeviceToHost, stream));
        bool dampFromBackup = false;
        while(true){
            typename solver::HessianSolver<Traits>::Threshold threshold{
                HessianBase<Traits>::EcBlock::ones() * 1E-3f,//@fixme: should not be absolute value. Use fixed relative error instead.
                HessianBase<Traits>::EaBlock::ones() * 1E-3f,
                HessianBase<Traits>::EbBlock::ones() * 1E-3f};
            dampHessian(dampFromBackup, mu, stream);
            dampFromBackup = true;
            const bool pcgConverged = hessianSolver->solve(threshold, maxIterPCG, stream);
            if (false && mVerbose && !pcgConverged) {
                std::cout << "  PCG did not converge after " << maxIterPCG << " iterations." << std::endl;
            }
            devModel->update(*devModelVarBackup, *delta, stream); // moved here to coordinate checkCondition2
            if (checkCondition2(stream)){
                devModel->revert(*devModelVarBackup, stream);
                stopReason += "update is too small compared to variables; ";
                stop = true;
            }
            else{
                checkCudaErrors(rba::dampLM::launchCudaComputeExpectedErrSqrNormDelta(
                        &devScalar[0], mu,
                        delta->getParamConst(), devHessianDiagBackup->getParamConst(), devHessian->getGConst(),
                        stream));
                float& expectedErrSqrNormDelta = hostScalar[0];
                checkCudaErrors(cudaMemcpyAsync(&expectedErrSqrNormDelta, &devScalar[0], sizeof(expectedErrSqrNormDelta), cudaMemcpyDeviceToHost, stream));

                // moved to before checkCondition2 as checkCondition2 requires it
//                devModel->update(*devModelVarBackup, *delta, stream);
                computeErrSqrNorm(stream);
                double& errSqrNormNew = *hostErrSqrNormNew;
                checkCudaErrors(cudaMemcpyAsync(&errSqrNormNew, devErrSqrNorm.get(), sizeof(*devErrSqrNorm), cudaMemcpyDeviceToHost, stream));
                syncStream(stream);
                if (std::isnan(errSqrNormNew)){
                    std::cout << format("  Got NAN%s.") % (mu > 1e4f ? ", but lambda is large and it might be OK" : "") << std::endl;
                    stopReason += "found NAN; ";
                    stop = true;
                }
                const float rho = (*hostErrSqrNormLast - errSqrNormNew) / expectedErrSqrNormDelta;
                if (rho > 0 && errSqrNormNew < *hostErrSqrNormLast){ // @fixme: (errSqrNormNew < *hostErrSqrNormLast) is not in the SBA paper. need validation
                    if (k + 1 < maxIters) {
                        const bool cond4 = checkCondition4(stream);
                        if (cond4) {
                            stopReason += "chi2 decrease is too small; ";
                        }
                        stop = stop || cond4;
                        computeHessian(false, stream);
                        const bool cond1 = checkCondition1(stream);
                        if (cond1) {
                            stopReason += "max(g) is too small; ";
                        }
                        stop = stop || checkCondition1(stream);
                    }
                    else{
                        stopReason += "reached max iterations; ";
                        stop = true;
                    }
                    mu *= std::max(1.f/3, 1.f - std::pow(2*rho-1, 3.f));
                    v = 2;
                    chi2 = errSqrNormNew;
                    nbAcceptedSteps++;
                    break;
                }
                else{
                    devModel->revert(*devModelVarBackup, stream);
                    nbRejectedSteps++;
                    if (mVerbose) {
                        std::cout << format("  Reverted LM step: chi2 = %e, lambda = %e") % errSqrNormNew % mu
                                  << std::endl;
                    }
                    mu *= v;
                    v *= 2;
                    if (mu > maxDamp){
                        stopReason += "reached max damp; ";
                        stop = true;
                        break;
                    }
                }
            }
            if (stop) {break;}
        }
        const bool cond3 = checkCondition3(stream);
        if (cond3){ stopReason += "error is too small"; }
        stop = stop || cond3;
        if (stop) {
            break;
        }
    }
    if (mVerbose) {
        std::cout << format("Result: chi2 = %e, lambda = %e with %u/%u accepted LM updates") % chi2 % mu %
                     nbAcceptedSteps % (nbAcceptedSteps + nbRejectedSteps) << std::endl;
        std::cout << "Stop reason: " << stopReason << std::endl;
    }
    devModel->saveToHost(*hostModel);
}

template <typename Traits>
void BundleSolver<Traits>::dampHessian(bool fromBackup, float damp, cudaStream_t stream) const {
    if (!fromBackup) {
        checkCudaErrors(dampLM::launchCudaBackupAndDamp(
                devHessianDiagBackup->getParam(), devHessian->getKernelArgs(), damp, stream));
    }
    else {
        checkCudaErrors(dampLM::launchCudaDampFromBackup(
                devHessianDiagBackup->getParamConst(), devHessian->getKernelArgs(), damp, stream));
    }
}

template <typename Traits>
bool BundleSolver<Traits>::checkCondition1(cudaStream_t stream) const {
    const HessianVec<Traits, true> gradient = {
            devHessian->Ec.data(), cast32u(devHessian->Ec.size()),
            devHessian->Ea.data(), cast32u(devHessian->Ea.size()),
            devHessian->Eb.data(), cast32u(devHessian->Eb.size())
    };
    checkCudaErrors(launchCudaGetAbsMax(devScalar.get(), gradient, stream));
    float& absMaxGrad = hostScalar[0];
    static_assert(std::is_same<decltype(&absMaxGrad), decltype(devScalar.get())>::value, "fatal error");
    checkCudaErrors(cudaMemcpyAsync(&absMaxGrad, devScalar.get(), sizeof(absMaxGrad), cudaMemcpyDeviceToHost, stream));
    syncStream(stream);
    return absMaxGrad < threshold1;
}

//@info: Note that we need to save model variables to devModelVarBackup before calling this
template <typename Traits>
bool BundleSolver<Traits>::checkCondition2(cudaStream_t stream) const
{
    const HessianVec<Traits, true> deltaVec = delta->getParamConst();
    checkCudaErrors(launchCudaGetSquaredNorm(&devScalar[0], deltaVec, stream));
    checkCudaErrors(cudaMemcpyAsync(&hostScalar[0], &devScalar[0], sizeof(devScalar[0]), cudaMemcpyDeviceToHost, stream));

    checkCudaErrors(launchCudaGetModelVarSquaredNorm(&devScalar[1], devModelVarBackup->template getParams<true>(), stream));
    checkCudaErrors(cudaMemcpyAsync(&hostScalar[1], devScalar.get(), sizeof(devScalar[1]), cudaMemcpyDeviceToHost, stream));

    syncStream(stream);

    return std::sqrt(hostScalar[0]) <= threshold2 * (std::sqrt(hostScalar[1]) + threshold2);
}

template <typename Traits>
bool BundleSolver<Traits>::checkCondition3(cudaStream_t stream) const {
    double& errSqrNorm = hostDoubleScalar[0];
    checkCudaErrors(cudaMemcpyAsync(&errSqrNorm, devErrSqrNorm.get(), sizeof(errSqrNorm), cudaMemcpyDeviceToHost, stream));
    syncStream(stream);
    return std::sqrt(errSqrNorm) < threshold3;
}

template <typename Traits>
bool BundleSolver<Traits>::checkCondition4(cudaStream_t stream) const {
    double& errSqrNorm = hostDoubleScalar[0];
    checkCudaErrors(cudaMemcpyAsync(&errSqrNorm, devErrSqrNorm.get(), sizeof(errSqrNorm), cudaMemcpyDeviceToHost, stream));
    syncStream(stream);
    const float errSqrNormLast = *hostErrSqrNormLast;
    return (std::sqrt(errSqrNormLast) - std::sqrt(errSqrNorm)) < (threshold4 * std::sqrt(errSqrNormLast));
}

#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template class BundleSolver<TRAITS>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES
}
}
