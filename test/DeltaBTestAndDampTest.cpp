//
// Created by yao on 8/12/18.
//


#include "TestModel.h"
#include "utils_test.h"
#include <boost/format.hpp>
#include "../GroupModelTypes.h"
#include "../blockSolvers/solveHessian.h"
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

template <typename Traits>
class DeltaBTest : public TestModel<Traits>{
public:
    RBA_TEST_IMPORT_TestModel_MEMBERS(Traits);
private:
#define USING_HESSIAN_TYPE(type) using type = typename rba::HessianBase<Traits>::type
    USING_HESSIAN_TYPE(EcBlock);
    USING_HESSIAN_TYPE(EaBlock);
    USING_HESSIAN_TYPE(EbBlock);
    USING_HESSIAN_TYPE(MBlock);
    USING_HESSIAN_TYPE(UBlock);
    USING_HESSIAN_TYPE(VBlock);
    USING_HESSIAN_TYPE(QBlock);
    USING_HESSIAN_TYPE(SBlock);
    USING_HESSIAN_TYPE(WBlock);
#undef USING_HESSIAN_TYPE
public:
    void testDeltaB();
protected:
    void SetUp() override;
    void computeHostReference() override;

    Eigen::VectorXd hostDeltaCA;
    Eigen::VectorXd refDeltaB;
    rba::grouped::DevSchurVec<Traits> devDeltaCA;
    MngVector<typename rba::HessianBase<Traits>::EbBlock> devDeltaB;
    Eigen::VectorXd cudaDeltaB;
};

template <typename Traits>
void DeltaBTest<Traits>::SetUp() {
    TestModel<Traits>::SetUp();
    computeHostJacobian();
    computeHostHessian();
    hostDeltaCA = Eigen::VectorXd::NullaryExpr(EcBlock::rows() * nbVarIntri + EaBlock::rows() * nbVarCaps, [this](){return this->dist(this->rng) * 1E-2f;});
    devDeltaCA.resize(cast32u(nbVarIntri), cast32u(nbVarCaps));


    //set up devHessian->M, devHessian->Ec and deltaC
    {
        const uint32_t rowOffset = 0;
        const uint32_t colOffset = 0;
        for (size_t i = 0; i < devHessian->M.size(); i++) {
            {
                MBlock tmp;
                toEigenMap(tmp) = hostHessian.template block<MBlock::rows(), MBlock::cols()>(
                        rowOffset + MBlock::rows() * i,
                        colOffset + MBlock::cols() * i).template cast<typename MBlock::ValType>();
                devHessian->M[i] = tmp;
            }
            {
                toEigenMap(devHessian->Ec[i]) = hostHessianEpsilon.template block<EcBlock::rows(), 1>(
                        rowOffset + EcBlock::rows() * i, 0).template cast<typename EcBlock::ValType>();
            }
            {
                toEigenMap(devDeltaCA.c[i]) = hostDeltaCA.block<EcBlock::rows(), 1>(
                        rowOffset  + EcBlock::rows() * i, 0).template cast<typename EcBlock::ValType>();
            }
        }
    }
    //set up deviceHessian.U
    {
        const size_t rowOffset = MBlock::rows() * devHessian->M.size();
        const size_t colOffset = MBlock::cols() * devHessian->M.size();
        for (size_t i = 0; i < devHessian->U.size(); i++) {
            {
                UBlock tmp;
                toEigenMap(tmp) = hostHessian.template block<UBlock::rows(), UBlock::cols()>(
                        rowOffset + UBlock::rows() * i,
                        colOffset + UBlock::cols() * i).template cast<typename UBlock::ValType>();
                devHessian->U[i] = tmp;
            }
            {
                EaBlock tmp;
                toEigenMap(tmp) = hostHessianEpsilon.template block<EaBlock::rows(), 1>(
                        rowOffset + EaBlock::rows() * i, 0).template cast<typename EaBlock::ValType>();
                devHessian->Ea[i] = tmp;
            }
            {
                toEigenMap(devDeltaCA.a[i]) = hostDeltaCA.block<EaBlock::rows(), 1>(
                        rowOffset + EaBlock::rows() * i, 0).template cast<typename EaBlock::ValType>();
            }
        }
    }
    //set up devHessian->V
    {
        const size_t rowOffset = MBlock::rows() * devHessian->M.size() + UBlock::rows() * devHessian->U.size();
        const size_t colOffset = MBlock::cols() * devHessian->M.size() + UBlock::cols() * devHessian->U.size();
        for (size_t i = 0; i < devHessian->V.size(); i++) {
            {
                VBlock tmp;
                toEigenMap(tmp) = hostHessian.template block<VBlock::rows(), VBlock::cols()>(
                        rowOffset + VBlock::rows() * i,
                        colOffset + VBlock::cols() * i).template cast<typename VBlock::ValType>();
                devHessian->V[i] = tmp;
            }
            {
                EbBlock tmp;
                toEigenMap(tmp) = hostHessianEpsilon.template block<EbBlock::rows(), 1>(
                        rowOffset + EbBlock::rows() * i, 0).template cast<typename EbBlock::ValType>();
                devHessian->Eb[i] = tmp;
            }
        }
    }
    //set up devHessian->Q
    {
        const size_t rowOffset = 0;
        const size_t colOffset = MBlock::cols() * devHessian->M.size();
        for (size_t i = 0; i < devHessian->Q.blocks.size(); i++){
            const auto row = devHessian->Q.row[i];
            if (~row != 0) {
                toEigenMap(devHessian->Q.blocks[i]) = hostHessian.template block<QBlock::rows(), QBlock::cols()>(
                        rowOffset + QBlock::rows() * row,
                        colOffset + QBlock::cols() * i).template cast<typename QBlock::ValType>();;
            }
            else
                devHessian->Q.blocks[i].assignScalar(NAN);
        }
    }
    //set up devHessian->S
    {
        const size_t rowOffset = 0;
        const size_t colOffset = MBlock::cols() * devHessian->M.size() + QBlock::cols() * devHessian->U.size();
        devHessian->S.assignData(hostHessian.block(rowOffset, colOffset, SBlock::rows() * devHessian->M.size(), SBlock::cols() * devHessian->V.size()));
    }
    //set up devHessian->W
    {
        const size_t rowOffset = MBlock::cols() * devHessian->M.size();
        const size_t colOffset = MBlock::cols() * devHessian->M.size() + QBlock::cols() * devHessian->U.size();
        devHessian->W.assignData(hostHessian.block(rowOffset, colOffset, WBlock::rows() * devHessian->U.size(), WBlock::cols() * devHessian->V.size()));
    }
}

template <typename Traits>
void DeltaBTest<Traits>::computeHostReference(){
    const auto lenCA = EcBlock::rows() * nbVarIntri + EaBlock::rows() * nbVarCaps;
    const auto lenB = EbBlock::rows() * nbVarPoints;
    refDeltaB = hostHessian.block(lenCA, lenCA, lenB, lenB).ldlt().solve(hostHessianEpsilon.bottomRows(lenB) - hostHessian.block(lenCA, 0, lenB, lenCA) * hostDeltaCA);
}

template <typename Traits>
void DeltaBTest<Traits>::testDeltaB() {
    computeHostReference();
    const auto hessian = devHessian->getKernelArgsConst();
    if (hessian.nbVarPts == 0)
        return;

    devDeltaB.resize(nbVarPoints);
    using namespace rba;
    const pcg::SchurVec<Traits, true> deltaCA = devDeltaCA.getParamConst();
    DevVecVec<uint16_t> sparseIdxW2idxLocalOb;
    grouped::toDevice_sparseIdxW2idxLocalOb(sparseIdxW2idxLocalOb, grouped::make_sparseIdxW2idxLocalOb(*devModel, *devHessian));
    checkCudaErrors(launchCudaComputeInvV<Traits>(devInvVBlocks.get(), hessian.V, hessian.nbVarPts, stream));
    checkCudaErrors(cudaMemcpyAsync(devDeltaB.data(), hessian.Eb, sizeof(devDeltaB[0])*hessian.nbVarPts, cudaMemcpyDeviceToDevice, stream));
    {
        // these two can run in parallel if use cuda graph
        assert(hessian.nbVarIntri == deltaCA.nbCBlocks);
        checkCudaErrors(solve_deltaB::launchCudaComputeSTDeltaC<Traits>(devDeltaB.data(), hessian.S, deltaCA.c, stream));
        assert(deltaCA.nbABlocks == hessian.nbVarCaps);
        checkCudaErrors(solve_deltaB::launchCudaComputeWTDeltaA(devDeltaB.data(), devModel->getKernelArgs(),
                                                                sparseIdxW2idxLocalOb.getConst(), deltaCA.a, hessian.W, stream));
    }
    checkCudaErrors(solve_deltaB::launchCudaComputeMVInplace<Traits>(devDeltaB.data(), devInvVBlocks.get(), hessian.nbVarPts, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    cudaDeltaB.resize(EbBlock::rows() * nbVarPoints);
    for (unsigned i = 0; i < nbVarPoints; i++)
        cudaDeltaB.block<EbBlock::rows(), 1>(EbBlock::rows() * i, 0) = toEigenMap(devDeltaB[i]).template cast<double>();

    EXPECT_TRUE(cudaDeltaB.allFinite());
    const auto absOrRatioDiff = ((cudaDeltaB - refDeltaB).array().abs() / refDeltaB.array().abs().max(refDeltaB.array().abs())).min((cudaDeltaB - refDeltaB).array().abs()).eval();
    EXPECT_LE(absOrRatioDiff.maxCoeff(), 0.01);
}

#define RBA_DEFINE_DeltaBTest(r, data, TRAITS)\
    using BOOST_PP_CAT(DeltaBTest, TRAITS) = DeltaBTest<rba::TRAITS>;\
    TEST_F(BOOST_PP_CAT(DeltaBTest, TRAITS), random)\
    {\
        testDeltaB();\
    }\
BOOST_PP_SEQ_FOR_EACH(RBA_DEFINE_DeltaBTest, data, ALL_TRAITS)
#undef RBA_DEFINE_DeltaBTest

template <typename Traits>
class DampTest : public DeltaBTest<Traits>
{
public:
    void testDamp();

protected:
    void computeHostReference() override;
    float damp = 0.f;
    Eigen::MatrixXd ref;
};
template <typename Traits>
void DampTest<Traits>::computeHostReference() {
    ref = DeltaBTest<Traits>::hostHessian;
    EXPECT_TRUE((DeltaBTest<Traits>::hostHessian.diagonal().array() >= 0).all());
//    ref.diagonal() = hostHessian.diagonal() * (1 + damp);
    ref.diagonal() = DeltaBTest<Traits>::hostHessian.diagonal().array() + damp * DeltaBTest<Traits>::hostHessian.diagonal().array().sqrt();
}
template <typename Traits>
void DampTest<Traits>::testDamp() {
    rba::grouped::DevHessianVec<Traits> backup;
    backup.resize(DeltaBTest<Traits>::nbVarIntri, DeltaBTest<Traits>::nbVarCaps, DeltaBTest<Traits>::nbVarPoints);

    for (int i = 0; i < 5; i++){
        damp = (this->dist(this->rng) + 0.5f) * 10.f;
        computeHostReference();
        if (i == 0) {
            checkCudaErrors(rba::dampLM::launchCudaBackupAndDamp(
                    backup.getParam(), this->devHessian->getKernelArgs(), damp, this->stream));
        }
        else {
            checkCudaErrors(rba::dampLM::launchCudaDampFromBackup(
                    backup.getParamConst(), this->devHessian->getKernelArgs(), damp, this->stream));
        }
        checkCudaErrors(cudaStreamSynchronize(this->stream));
        auto dampedHessian = DeviceHessian2Eigen(*this->devHessian);
        EXPECT_TRUE(dampedHessian.allFinite());
        const auto absOrRatioDiff = ((dampedHessian - ref).array().abs() / ref.array().abs().max(ref.array().abs())).min((dampedHessian - ref).array().abs()).eval();
        EXPECT_LE(absOrRatioDiff.maxCoeff(), 0.01);

//        std::cout << boost::format("Diff Ratio (max = %d) *************************************************************************************\n") % absOrRatioDiff.maxCoeff();
//        std::cout << absOrRatioDiff << "\n\n";
//        std::cout << std::flush;
//
//        std::cout << "Damped *******************************************************************************************\n";
//        std::cout << dampedHessian << "\n\n";
//
//        std::cout << "REF *******************************************************************************************\n";
//        std::cout << ref << "\n\n";
//
//        std::cout << "Orig *******************************************************************************************\n";
//        std::cout << hostHessian << "\n\n";
    }
}

#define RBA_DEFINE_DampTest(r, data, TRAITS)\
    using BOOST_PP_CAT(DampTest_, TRAITS) = DampTest<rba::TRAITS>;\
    TEST_F(BOOST_PP_CAT(DampTest_, TRAITS), random)\
    {\
        testDamp();\
    }
BOOST_PP_SEQ_FOR_EACH(RBA_DEFINE_DampTest, data, ALL_TRAITS)
#undef RBA_DEFINE_DampTest
