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
// Created by yao on 27/10/18.
//

#include "TestModel.h"
#include "utils_test.h"
#include <boost/format.hpp>
#include "../GroupModelTypes.h"
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

template <typename Traits>
class HessianTest : public TestModel<Traits>{
public:
    RBA_IMPORT_TRAITS(Traits);
    RBA_TEST_IMPORT_TestModel_MEMBERS(Traits);
    void testHessian();
    void SetUp() override{
        TestModel<Traits>::SetUp();
    }
protected:
    void computeHessian();
    void computeHostReference() override{
        computeHostJacobian();
        computeHostHessian();
    }

    Eigen::MatrixXd cudaHessian;
    Eigen::VectorXd cudaEpsilon;
};

template <typename Traits>
void HessianTest<Traits>::computeHessian() {
    devModel->init(hostModel);
    //@fixme: add test for enableWBlock = false
    devHessian->init(hostModel, devModel->involvedCaps.data(), devModel->involvedCaps.size(), devHessianPreComp.get(), true);

    checkCudaErrors(rba::launchCudaComputeHessian(devModel->getKernelArgs(), nullptr, devHessian->getKernelArgs(),
            cast32u(devHessian->S.data.size()), devHessianPreComp->getKernelArgs(), stream));

    checkCudaErrors(cudaStreamSynchronize(stream));

    auto cudaResults = DeviceHessianAndEpsilon2Eigen(*devHessian);
    cudaHessian = cudaResults.first;
    cudaEpsilon = cudaResults.second;

//    std::cout << "Diff *******************************************************************************************\n";
//    std::cout << cudaHessian - hostHessian << "\n\n";
//
//    const auto absOrRatioDiff = ((cudaHessian - hostHessian).array().abs() / cudaHessian.array().abs().max(hostHessian.array().abs())).min((cudaHessian - hostHessian).array().abs()).eval();
//    std::cout << boost::format("Diff Ratio (max = %d) *************************************************************************************\n") % absOrRatioDiff.maxCoeff();
//    std::cout << absOrRatioDiff << "\n\n";
//    std::cout << std::flush;
//
//    std::cout << "Hessian *******************************************************************************************\n";
//    std::cout << cudaHessian << "\n\n";

}

template <typename Traits>
void HessianTest<Traits>::testHessian() {
    computeHostReference();
    computeHessian();
    EXPECT_TRUE((((cudaHessian - hostHessian).array().abs() / cudaHessian.array().abs().max(hostHessian.array().abs()).max(1E-6f)).array() < 0.1f || ((cudaHessian - hostHessian).array().abs()).array() < 1.f).all());
//    const auto absOrRatioDiff = ((cudaHessian - hostHessian).array().abs() / cudaHessian.array().abs().max(hostHessian.array().abs())).min((cudaHessian - hostHessian).array().abs()).eval();
//    EXPECT_LE(absOrRatioDiff.maxCoeff(), 0.3);

    const auto absOrRatioDiffEpsilon = ((cudaEpsilon - hostHessianEpsilon).array().abs() / cudaEpsilon.array().abs().max(hostHessianEpsilon.array().abs())).min((cudaEpsilon - hostHessianEpsilon).array().abs()).eval();
    EXPECT_LE(absOrRatioDiffEpsilon.maxCoeff(), 0.1);
}

#define RBA_DEFINE_HessianTest(r, data, TRAITS)\
    using BOOST_PP_CAT(HessianTest_, TRAITS) = HessianTest<rba::TRAITS>;\
    TEST_F(BOOST_PP_CAT(HessianTest_, TRAITS), random)\
    {\
        testHessian();\
    }
BOOST_PP_SEQ_FOR_EACH(RBA_DEFINE_HessianTest, data, ALL_TRAITS)
#undef RBA_DEFINE_HessianTest

