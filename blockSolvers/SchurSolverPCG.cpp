//
// Created by yao on 2/12/18.
//

#include "SchurSolverPCG.h"
#include "solveSchurPCG.h"
#include <boost/preprocessor/seq/for_each.hpp>

namespace {
template <typename T>
cudaError_t cudaSetScalarZero(T& scalar, cudaStream_t stream){
    return cudaMemsetAsync(&scalar, 0, sizeof(scalar), stream);
}
}
namespace rba{
namespace solver{

template <typename Traits>
void SchurSolverPCG<Traits>::setUp(const rba::grouped::DeviceSchur<Traits> &devA, const pcg::SchurVec<Traits, false> &x, cudaStream_t stream) {
    //@info: what we do here should only depend on structure of A and b, but not the content of them
    pcgPreComp.init(devA);
    checkCudaErrors(devA.migrateToDevice(getCudaDevice(), stream));
    Schur<Traits, true> A = devA.getKernelArgsConst();

    this->schurA = A;
    this->schurX = x;
    this->b = pcg::SchurVec<Traits, true>{A.Ec, A.nbVarIntri, A.Ea, A.nbVarCaps};

    preCond = {A.diagM, A.nbVarIntri, A.diagU, A.nbVarCaps};
    invPreCond.resize(A.nbVarIntri, A.nbVarCaps);
    r.resize(A.nbVarIntri, A.nbVarCaps);
    z.resize(A.nbVarIntri, A.nbVarCaps);
    p.resize(A.nbVarIntri, A.nbVarCaps);
    if (devScalars == nullptr) {
        devScalars = deviceAlloc<PCGScalars>(1);
        static const epf tmpScalar = 1.0;
        checkCudaErrors(cudaMemcpyAsync(&devScalars->one, &tmpScalar, sizeof(tmpScalar), cudaMemcpyHostToDevice, stream));
    }
    tmp.resize(A.nbVarIntri, A.nbVarCaps);
    if (hostCounter == nullptr) {
        hostCounter = hostAlloc<uint32_t>(1);
    }
    checkCudaErrors(migrateToDevice(getCudaDevice(), stream));
}

template <typename Traits>
void SchurSolverPCG<Traits>::prepare(cudaStream_t stream) {
    if (preCond.M->size() != 0) {
        checkCudaErrors(pcg::launchCudaComputeInverse(invPreCond.M.data(), preCond.M, preCond.nbMBlocks, stream));
    }
    if (preCond.U->size() != 0) {
        checkCudaErrors(pcg::launchCudaComputeInverse(invPreCond.U.data(), preCond.U, preCond.nbUBlocks, stream));
    }
#if RBA_SANITY_CHECK
    checkCudaErrors(invPreCond.migrateToDevice(cudaCpuDeviceId, stream));
    syncStream(stream);
    for (uint32_t i = 0; i < invPreCond.M.size(); i++){
        if (!invPreCond.M.at(i).allFinite()){
            throw std::runtime_error("bad floating-point value");
        }
    }
    for (uint32_t i = 0; i < invPreCond.U.size(); i++){
        if (!invPreCond.U.at(i).allFinite()){
            throw std::runtime_error("bad floating-point value");
        }
    }
    checkCudaErrors(invPreCond.migrateToDevice(getCudaDevice(), stream));
#endif
    idxIter = 0;
}

template <typename Traits>
void SchurSolverPCG<Traits>::computeR(cudaStream_t stream) {
    auto& Ax = tmp;
    computeMMV(Ax.getParam(), schurX, stream);
    checkCudaErrors(pcg::SchurVecAdd::launch<Traits>(r.getParam(), &devScalars->one, true, Ax.getParam(), b, stream));
}

template <typename Traits>
static pcg::SchurDiag<Traits, true> getSchurDiag(const Schur<Traits, true> &A){
    return {A.diagM, A.nbVarIntri, A.diagU, A.nbVarCaps};
}

template <typename Traits>
void SchurSolverPCG<Traits>::computeMMV(const rba::pcg::SchurVec<Traits, false> &dst,
                                const rba::pcg::SchurVec<Traits, true> &src, cudaStream_t stream) {
    //this kernel also works as initialization for Ap.
    checkCudaErrors(pcg::SchurMMV::launchForDiag(dst, nullptr, getSchurDiag(schurA), src, {}, stream));
    //the follow kernels can be in multiple streams
    {
        if constexpr(isGroupModel<Traits>()) {
            if (schurA.upperM.nbRows != 0) {
                checkCudaErrors(pcg::SchurMMV::launchForUpperM<Traits>(dst.c, schurA.upperM, src.c, stream));
            }
            if (schurA.upperM.nbRows != 0) {
                checkCudaErrors(
                        pcg::SchurMMV::launchForLowerM<Traits>(dst.c, pcgPreComp.lowerM.getConst(), schurA.upperM,
                                                               src.c, stream));
            }
            if (schurA.Q.nbRows != 0) {
                checkCudaErrors(pcg::SchurMMV::launchForUpperQ<Traits>(dst.c, schurA.Q, src.a, stream));
            }
            if (schurA.Q.nbRows != 0) {
                checkCudaErrors(
                        pcg::SchurMMV::launchForLowerQ<Traits>(dst.a, pcgPreComp.lowerQ.getConst(), schurA.Q, src.c,
                                                               stream));
            }
        }
        checkCudaErrors(pcg::SchurMMV::launchForUpperU<Traits>(dst.a, schurA.upperU, src.a, stream));
        checkCudaErrors(pcg::SchurMMV::launchForLowerU<Traits>(dst.a, pcgPreComp.lowerU.getConst(), schurA.upperU, src.a, stream));
    }
}

template <typename Traits>
void SchurSolverPCG<Traits>::iterate(cudaStream_t stream) {
    if(idxIter % nItersRestart == 0){
        computeR(stream);
        auto& z0 = p;
        checkCudaErrors(cudaSetScalarZero(devScalars->zTr[0], stream));
        //@fixme: replace invPreCond with LLT solver
        checkCudaErrors(pcg::SchurMMV::launchForDiag<Traits>(z0.getParam(), &devScalars->zTr[0], invPreCond.getParamConst(), r.getParam(), r.getParam(), stream));
    }
    else{
        checkCudaErrors(cudaSetScalarZero(devScalars->zTr[1], stream));
        checkCudaErrors(
                pcg::SchurMMV::launchForDiag<Traits>(z.getParam(), &devScalars->zTr[1], invPreCond.getParamConst(), r.getParam(),
                                             r.getParam(), stream));
        checkCudaErrors(pcg::launchUpdateBeta(devScalars->beta, devScalars->zTr, stream));
        // The underline kernel for SchurVecAdd::launch is not using __restrict__ in kernel arguments, to allow aliasing
        checkCudaErrors(pcg::SchurVecAdd::launch<Traits>(p.getParam(), &devScalars->beta, false, p.getParam(), z.getParamConst(), stream));
    }
    auto& Ap = tmp;
    computeMMV(Ap.getParam(), p.getParamConst(), stream);
    checkCudaErrors(pcg::SchurDotProd::launch(&devScalars->pTAp, p.getParamConst(), Ap.getParamConst(), stream));
    checkCudaErrors(pcg::launchUpdateAlpha(devScalars->alpha, devScalars->zTr[0], devScalars->pTAp, stream));
    checkCudaErrors(pcg::SchurVecAdd::launch<Traits>(schurX, &devScalars->alpha, false, p.getParamConst(), schurX, stream));
    idxIter++;
    if (idxIter % nItersRestart != 0 || idxIter % intervalCheckConvergence == 0)
        checkCudaErrors(pcg::SchurVecAdd::launch(r.getParam(), &devScalars->alpha, true, Ap.getParamConst(), r.getParamConst(), stream));
}

template <typename Traits>
uint32_t SchurSolverPCG<Traits>::checkConvergenceSync(const Threshold &threshold, cudaStream_t stream) {
    checkCudaErrors(cudaSetScalarZero(devScalars->counter, stream));
    checkCudaErrors(pcg::CheckThreshold::launch(&devScalars->counter, r.getParamConst(), threshold.c, threshold.a, stream));
    checkCudaErrors(cudaMemcpyAsync(hostCounter.get(), &devScalars->counter, sizeof(devScalars->counter), cudaMemcpyDeviceToHost, stream));
    // @fixme: sometimes program is stuck here, in libcuda.so. Likely a cuda driver bug.
    // @fixme: Maybe it's driver bug related to managed memory?
    // @fixme: Looks like it won't reproduce on Turing. Maybe a hardware bug?
    syncStream(stream);
    return *hostCounter;
}

template <typename Traits>
bool SchurSolverPCG<Traits>::solve(const Threshold &threshold, uint32_t maxIters, cudaStream_t stream) {
    prepare(stream);
    while (idxIter < maxIters){
        iterate(stream);
        if (idxIter % intervalCheckConvergence == 0){
            const uint32_t nbViolation = checkConvergenceSync(threshold, stream);
            if (nbViolation == 0)
                return true;
        }
    }
    return false;
}

template <typename Traits>
cudaError_t SchurSolverPCG<Traits>::migrateToDevice(int deviceId, cudaStream_t stream) const {
    checkCudaErrors(pcgPreComp.migrateToDevice(deviceId, stream));
    checkCudaErrors(invPreCond.migrateToDevice(deviceId, stream));
    checkCudaErrors(r.migrateToDevice(deviceId, stream));
    checkCudaErrors(z.migrateToDevice(deviceId, stream));
    checkCudaErrors(p.migrateToDevice(deviceId, stream));
    checkCudaErrors(tmp.migrateToDevice(deviceId, stream));
    return cudaSuccess;
}

#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template class SchurSolverPCG<TRAITS>;
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES
}
}

