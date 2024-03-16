//
// Created by yao on 2/12/18.
//

#pragma once

#include "../GroupModelTypes.h"
#include "solveSchurPCG.h"
#include "SchurSolver.h"

namespace rba{
namespace solver{

// https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
template <typename Traits>
class SchurSolverPCG : public SchurSolver<Traits>{
    RBA_IMPORT_TRAITS(Traits);
    using typename SchurSolver<Traits>::Threshold;
public:
    // set up space for solving. A and b do not need to contain the data used for solving but should have correct structure.
    // the pointers inside A and x need to be valid during solving.
    void setUp(const rba::grouped::DeviceSchur<Traits> &devA, const pcg::SchurVec<Traits, false> &x, cudaStream_t stream) override;
    bool solve(const Threshold &threshold, uint32_t maxIters, cudaStream_t stream) override;
protected:
    // First step inside solve(). set up some data that persists across iterations.
    void prepare(cudaStream_t stream);
    void iterate(cudaStream_t stream);
    uint32_t checkConvergenceSync(const Threshold &threshold, cudaStream_t stream);

    // re-compute r from (b-Ax) rather than (r-alpha*Ap), which accumulates numeric error and lose orthogonality
    void computeR(cudaStream_t stream);
    // compute Schur * src
    virtual void computeMMV(const rba::pcg::SchurVec<Traits, false> &dst, const rba::pcg::SchurVec<Traits, true> &src, cudaStream_t stream);

    cudaError_t migrateToDevice(int deviceId, cudaStream_t stream) const;
private:
    Schur<Traits, true> schurA = {};
    grouped::DevicePCGPreComp pcgPreComp = {};
protected:
    pcg::SchurVec<Traits, false> schurX = {};
    pcg::SchurVec<Traits, true> b = {};
    pcg::SchurDiag<Traits, true> preCond = {}; // pre-conditioner
    grouped::DevSchurDiag<Traits> invPreCond; //inversion of Jacobi preconditioner
    grouped::DevSchurVec<Traits> r;
    grouped::DevSchurVec<Traits> z;
    grouped::DevSchurVec<Traits> p;
    struct PCGScalars{
        epf beta;
        epf zTr[2];//z.T*r
        epf pTAp;//p.T*A*p
        epf alpha;
        epf one;//contains value 1.0
        uint32_t counter;//count number of residue blocks violating threshold
    };
    std::unique_ptr<PCGScalars, CudaDeviceDeleter> devScalars;
    uint32_t idxIter = 0;

    // after some iterations, restart PCG to recover orthogonality
    uint32_t nItersRestart = 64;

    grouped::DevSchurVec<Traits> tmp;

    std::unique_ptr<uint32_t, CudaHostDeleter> hostCounter; //host part for devScalars->counter
    uint32_t intervalCheckConvergence = 4; // interval for convergence checking in number of iterations

    void syncStream(cudaStream_t stream) const{
#if 1
        checkCudaErrors(cudaStreamSynchronize(stream));
#else
        // lower chance to deadlock, but still can happen
        while (true) {
            const auto err = cudaStreamQuery(stream);
            if (err == cudaSuccess){
                break;
            }
            else if (err == cudaErrorNotReady) {
                continue;
            }
            checkCudaErrors(err);
        }
#endif
    }
};

}
}



