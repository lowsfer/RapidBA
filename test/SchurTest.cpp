//
// Created by yao on 18/11/18.
//


#include "TestModel.h"
#include "utils_test.h"
#include <boost/format.hpp>
#include "../computeSchur.h"
#include "../GroupModelTypes.h"
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

template <typename Traits>
class SchurTest : public TestModel<Traits>{
    RBA_TEST_IMPORT_TestModel_MEMBERS(Traits);
public:
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
    void testSchur();
protected:
    void SetUp() override;
    void computeHostReference() override{
        computeHostSchur();
    }
    void computeSchur();
    Eigen::MatrixXd cudaSchur;
    Eigen::VectorXd cudaSchurEpsilon;
};

template <typename Traits>
void SchurTest<Traits>::SetUp() {
    TestModel<Traits>::SetUp();
    computeHostJacobian();
    computeHostHessian();
    //set up devHessian->M and devHessian->Ec
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
                EcBlock tmp;
                toEigenMap(tmp) = hostHessianEpsilon.template block<EcBlock::rows(), 1>(
                        rowOffset + EcBlock::rows() * i, 0).template cast<typename EcBlock::ValType>();
                devHessian->Ec[i] = tmp;
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
void SchurTest<Traits>::computeSchur() {
    devSchur->init(*devModel, *devHessian, devSchurPreComp.get(), nullptr);
    checkCudaErrors(rba::launchCudaComputeInvV<Traits>(devInvVBlocks.get(), devHessian->getKernelArgsConst().V,
                                               devHessian->getKernelArgsConst().nbVarPts, stream));
    checkCudaErrors(rba::launchCudaComputeSchur(devModel->getKernelArgs(), devHessian->getKernelArgsConst(),
                                                devInvVBlocks.get(), devSchur->getKernelArgs(),
                                                devSchurPreComp->toParams(),
                                                stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    cudaSchur = DeviceSchur2Eigen(*devSchur);
    cudaSchurEpsilon = DeviceSchurEpsilon2Eigen(*devSchur);
}

template <typename Traits>
void SchurTest<Traits>::testSchur() {
    computeHostReference();
    computeSchur();
    EXPECT_TRUE(cudaSchur.allFinite());
    const auto absOrRatioDiff = ((cudaSchur - hostSchur).array().abs() / hostSchur.array().abs().max(hostSchur.array().abs()).max(1E-6f)).min((cudaSchur - hostSchur).array().abs()).eval();
    EXPECT_LE(absOrRatioDiff.maxCoeff(), 0.02);
    const auto absOrRatioDiffEpsilon = ((cudaSchurEpsilon - hostSchurEpsilon).array().abs() / hostSchurEpsilon.array().abs().max(hostSchurEpsilon.array().abs())).min((cudaSchurEpsilon - hostSchurEpsilon).array().abs()).eval();
    EXPECT_LE(absOrRatioDiffEpsilon.maxCoeff(), 0.01);

    //std::cerr << "[**********] Epsilon test is not implemented\n";

//    std::cout << "Diff *******************************************************************************************\n";
//    std::cout << cudaSchur - hostSchur << "\n\n";
//
//    std::cout << boost::format("Diff Ratio (max = %d) *************************************************************************************\n") % absOrRatioDiff.maxCoeff();
//    std::cout << absOrRatioDiff << "\n\n";
//    std::cout << std::flush;
//
//    std::cout << "Schur *******************************************************************************************\n";
//    std::cout << cudaSchur << "\n\n";
//
//    std::cout << "Host Schur *******************************************************************************************\n";
//    std::cout << hostSchur << "\n\n";

//    Eigen::Matrix<double, -1, 2> epsilon(hostSchurEpsilon.rows(), 2);
//    epsilon.col(0) = cudaSchurEpsilon;
//    epsilon.col(1) = hostSchurEpsilon;
//    std::cout << "cuda vs host Schur epsilon**********************************************************\n";
//    std::cout << epsilon << "\n\n";
}

#define RBA_DEFINE_SchurTest(r, data, TRAITS)\
    using BOOST_PP_CAT(SchurTest_, TRAITS) = SchurTest<rba::TRAITS>;\
    TEST_F(BOOST_PP_CAT(SchurTest_, TRAITS), random)\
    {\
        testSchur();\
    }
BOOST_PP_SEQ_FOR_EACH(RBA_DEFINE_SchurTest, data, ALL_TRAITS)
#undef RBA_DEFINE_SchurTest
