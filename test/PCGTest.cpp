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
// Created by yao on 4/12/18.
//

#include "TestModel.h"
#include "utils_test.h"
#include <boost/format.hpp>
#include "../GroupModelTypes.h"
#include "../kmat.h"
#include "../utils_host.h"
#include "../blockSolvers/SchurSolverPCG.h"
#include <fstream>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

template <typename Traits>
class PCGUtilTest : public testing::Test{
public:
    RBA_IMPORT_TRAITS(Traits);
    void SetUp() override{
        auto init = [this](rba::grouped::DevSchurVec<Traits>& vec)->void {
            vec.resize(sizeC, sizeA);
            for(auto& elem : vec.c)
                toEigenMap(elem) = Eigen::Matrix<epf, CamIntr::DoF, 1>::NullaryExpr([this]() -> double { return dist(rng); });
            for(auto& elem : vec.a)
                toEigenMap(elem) = Eigen::Matrix<hpf, Capture::DoF, 1>::NullaryExpr([this]() -> float { return dist(rng); });
        };
        init(x);
        init(y);
    }
    void testDotProd() const {
        const double ref = [this] {
            double result = 0;
            for (uint32_t i = 0; i < sizeC; i++)
                result += toEigenMap(x.c[i]).dot(toEigenMap(y.c[i]));
            for (uint32_t i = 0; i < sizeA; i++)
                result += toEigenMap(x.a[i]).dot(toEigenMap(y.a[i]));
            return result;
        }();
        const auto dst = managedAlloc<double>(1);
        *dst = 0;
        checkCudaErrors(rba::pcg::SchurDotProd::launch(dst.get(), x.getParamConst(), y.getParamConst(), stream));
        checkCudaErrors(cudaStreamSynchronize(stream));
        EXPECT_LE(absOrRelativeDiff(ref, *dst), 1E-3);
    }
    void testVecAdd() const{
        auto alpha = managedAlloc<double>(1);
        *alpha = dist(rng);
        const bool negativeAlpha = (dist(rng) > 0);
        rba::grouped::DevSchurVec<Traits> ref;
        ref.resize(sizeC, sizeA);
        for (uint32_t i = 0; i < sizeC; i++)
            ref.c[i] = x.c[i] * *alpha * (negativeAlpha ? -1 : 1) + y.c[i];
        for (uint32_t i = 0; i < sizeA; i++)
            ref.a[i] = x.a[i] * *alpha * (negativeAlpha ? -1 : 1) + y.a[i];
        rba::grouped::DevSchurVec<Traits> yCopy = y;
        checkCudaErrors(rba::pcg::SchurVecAdd::launch(yCopy.getParam(), alpha.get(), negativeAlpha, x.getParamConst(), yCopy.getParamConst(), stream));
        checkCudaErrors(cudaStreamSynchronize(stream));
        for (uint32_t i = 0; i < sizeC; i++)
            CHECK_KMAT_CLOSE(ref.c[i], yCopy.c[i], 1E-4f, seed);
        for (uint32_t i = 0; i < sizeA; i++)
            CHECK_KMAT_CLOSE(ref.a[i], yCopy.a[i], 1E-4f, seed);
    }
    void testCheckThreshold() const{
        using EcBlock = typename rba::HessianBase<Traits>::EcBlock;
        using EaBlock = typename rba::HessianBase<Traits>::EaBlock;
        EcBlock thresC; EaBlock thresA;
        std::fill_n(thresC.data(), thresC.size(), 0.5);
        std::fill_n(thresA.data(), thresA.size(), 0.5f);
        const uint32_t ref = uint32_t(std::count_if(x.c.begin(), x.c.end(), [thresC](EcBlock e){return !(toEigenMap(e).array().abs() <= toEigenMap(thresC).array()).all();})
                + std::count_if(x.a.begin(), x.a.end(), [thresA](EaBlock e){return !(toEigenMap(e).array().abs() <= toEigenMap(thresA).array()).all();}));
        auto count = managedAlloc<uint32_t>(1);
        *count = 0;
        checkCudaErrors(rba::pcg::CheckThreshold::launch(count.get(), x.getParamConst(), thresC, thresA, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));
        EXPECT_EQ(ref, *count);
    }

    cudaStream_t stream = nullptr;
    const uint32_t sizeC = 3;
    const uint32_t sizeA = 38;
    const uint32_t seed = 0;
    mutable std::default_random_engine rng = std::default_random_engine{seed};
    mutable std::uniform_real_distribution<float> dist = std::uniform_real_distribution<float>{-1.f, 1.f};
    rba::grouped::DevSchurVec<Traits> x;
    rba::grouped::DevSchurVec<Traits> y;
};

using PCGUtilTestGrpF2C2D5 = PCGUtilTest<rba::TraitsGrpF2C2D5Global>;
TEST_F(PCGUtilTestGrpF2C2D5, DotProd)
{
    testDotProd();
}
TEST_F(PCGUtilTestGrpF2C2D5, VecAdd)
{
    testVecAdd();
}
TEST_F(PCGUtilTestGrpF2C2D5, checkThreshold)
{
    testCheckThreshold();
}

using PCGUtilTestGrpF1D2 = PCGUtilTest<rba::TraitsGrpF1D2Global>;
TEST_F(PCGUtilTestGrpF1D2, DotProd)
{
    testDotProd();
}
TEST_F(PCGUtilTestGrpF1D2, VecAdd)
{
    testVecAdd();
}
TEST_F(PCGUtilTestGrpF1D2, checkThreshold)
{
    testCheckThreshold();
}

template <typename Traits>
class PCGTest : public testing::Test, public rba::solver::SchurSolverPCG<Traits>{
public:
    void testMMV();
    void testComputeR();
    void testSolve();
protected:
    void SetUp() override;
    void dumpProblem();
private:
    template <typename KMatType>
    KMatType genRand(){
        KMatType result;
        std::generate_n(result.data(), result.size(), [this]()->typename KMatType::ValType{return dist(rng);});
        return result;
    }

    cudaStream_t stream = nullptr;
    const uint32_t sizeC = rba::isGroupModel<Traits>() ? 3 : 0;
    const uint32_t sizeA = 38;
    const uint32_t seed = 0;
    mutable std::default_random_engine rng = std::default_random_engine{seed};
    mutable std::uniform_real_distribution<float> dist = std::uniform_real_distribution<float>{-.5f, .5f};
    mutable std::binomial_distribution<int> distRemove = std::binomial_distribution<int>(1, 0.95f);
    rba::grouped::DeviceSchur<Traits> schur;
    rba::grouped::DevSchurVec<Traits> x0;
    rba::grouped::DevSchurVec<Traits> x;
};

template <typename Traits>
void PCGTest<Traits>::SetUp() {
    testing::Test::SetUp();
#define USING_HESSIAN_TYPE(type) using type = typename rba::HessianBase<Traits>::type
    USING_HESSIAN_TYPE(EcBlock);
    USING_HESSIAN_TYPE(EaBlock);
    USING_HESSIAN_TYPE(EbBlock);
    USING_HESSIAN_TYPE(QBlock);
#undef USING_HESSIAN_TYPE
    auto initSchurVec = [this](rba::grouped::DevSchurVec<Traits>& vec)->void{
        vec.resize(sizeC, sizeA);
        std::generate(vec.c.begin(), vec.c.end(), [this]{return genRand<EcBlock>();});
        std::generate(vec.a.begin(), vec.a.end(), [this]{return genRand<EaBlock>();});
    };
    initSchurVec(x0);
    initSchurVec(x);

    //initialize schur
    auto initPositiveDefinite = [this](auto& vec, uint32_t size)->void{
        vec.resize(size);
        using SymBlockType = typename std::remove_reference_t<decltype(vec)>::value_type;
        std::generate(vec.begin(), vec.end(), [this]{
            return genPositiveDefiniteSymKMat<SymBlockType>([this](){return dist(rng);}, sizeC*EcBlock::rows()+sizeA*EaBlock::rows());
        });
    };
    initPositiveDefinite(schur.diagM, sizeC);
    initPositiveDefinite(schur.diagU, sizeA);
    auto initUpper = [this](auto& upper, uint32_t count)->void{
        for (uint32_t i = 0; i < count; i++){
            using BlockType = typename std::remove_reference<decltype(upper.data[0])>::type;
            const uint32_t rowLength = count - i - 1;
            std::vector<uint32_t> idxCol(rowLength);
            std::iota(idxCol.begin(), idxCol.end(), i+1);
            std::shuffle(idxCol.begin(), idxCol.end(), rng);
            idxCol.resize(rowLength / 2);
            std::sort(idxCol.begin(), idxCol.end());
//            idxCol.erase(std::remove_if(idxCol.begin(), idxCol.end(), [this](uint32_t){return distRemove(rng);}), idxCol.end());
            assert(std::is_sorted(idxCol.begin(), idxCol.end()));
            std::vector<BlockType> blocks(idxCol.size());
            std::generate(blocks.begin(), blocks.end(), [this](){return genRand<BlockType>();});
            upper.nbRows++;
            upper.rows.emplace_back(upper.rows.back() + cast32u(idxCol.size()));
            assert(upper.nbRows + 1 == upper.rows.size());
            upper.idxCol.insert(upper.idxCol.end(), idxCol.begin(), idxCol.end());
            upper.data.insert(upper.data.end(), blocks.begin(), blocks.end());
            assert(upper.rows.back() == upper.idxCol.size());
            assert(upper.rows.back() == upper.data.size());
        }
    };
    initUpper(schur.upperM, sizeC);
    initUpper(schur.upperU, sizeA);
    for (uint32_t i = 0; i < sizeC; i++){
        using BlockType = QBlock;
        const uint32_t rowLength = sizeA;
        std::vector<uint32_t> idxCol(rowLength);
        std::iota(idxCol.begin(), idxCol.end(), 0);
        std::shuffle(idxCol.begin(), idxCol.end(), rng);
        idxCol.resize(rowLength / 2);
        std::sort(idxCol.begin(), idxCol.end());
//        idxCol.erase(std::remove_if(idxCol.begin(), idxCol.end(), [this](uint32_t){return distRemove(rng);}), idxCol.end());
        assert(std::is_sorted(idxCol.begin(), idxCol.end()));
        std::vector<BlockType> blocks(idxCol.size());
        std::generate(blocks.begin(), blocks.end(), [this](){return genRand<BlockType>();});
        auto& Q = schur.Q;
        Q.nbRows++;
        Q.rows.emplace_back(Q.rows.back() + idxCol.size());
        assert(Q.nbRows + 1 == Q.rows.size());
        Q.idxCol.insert(Q.idxCol.end(), idxCol.begin(), idxCol.end());
        Q.data.insert(Q.data.end(), blocks.begin(), blocks.end());
        assert(Q.rows.back() == Q.idxCol.size());
        assert(Q.rows.back() == Q.data.size());
    }
    auto initVec = [this](auto& vec, uint32_t size)->void{
        vec.resize(size);
        using BlockType = typename std::remove_reference_t<decltype(vec)>::value_type;
        std::generate(vec.begin(), vec.end(), [this]{return genRand<BlockType>();});
    };
    initVec(schur.Ec, sizeC);
    initVec(schur.Ea, sizeA);
    rba::solver::SchurSolverPCG<Traits>::setUp(schur, x.getParam(), stream);
    this->computeMMV(rba::pcg::SchurVec<Traits, false>{schur.Ec.data(), sizeC, schur.Ea.data(), sizeA}, x0.getParamConst(), stream);
}

template <typename Traits>
void PCGTest<Traits>::testMMV() {
    rba::grouped::DevSchurVec<Traits> dst;
    dst.resize(sizeC, sizeA);
    this->computeMMV(dst.getParam(), x0.getParamConst(), stream);
    checkCudaErrors(cudaStreamSynchronize(stream));
    const Eigen::VectorXd result = SchurVec2Eigen(dst.getParamConst());
    const Eigen::VectorXd ref = DeviceSchur2Eigen(schur) * SchurVec2Eigen(x0.getParamConst());
    EXPECT_TRUE(((result - ref).array().abs() < 1E-3f).all());
}
template <typename Traits>
void PCGTest<Traits>::testComputeR() {
    this->computeR(stream);
    checkCudaErrors(cudaStreamSynchronize(stream));
    const Eigen::VectorXd result = SchurVec2Eigen(this->r.getParamConst());
    const Eigen::VectorXd ref = SchurVec2Eigen(this->b) - DeviceSchur2Eigen(schur) * SchurVec2Eigen(x.getParamConst());
    EXPECT_TRUE(((result - ref).array().abs() < 1E-3f).all());
}
template <typename Traits>
void PCGTest<Traits>::testSolve() {
    //dumpProblem();
#define USING_HESSIAN_TYPE(type) using type = typename rba::HessianBase<Traits>::type
    USING_HESSIAN_TYPE(EcBlock);
    USING_HESSIAN_TYPE(EaBlock);
#undef USING_HESSIAN_TYPE
    this->solve(typename rba::solver::SchurSolver<Traits>::Threshold{EcBlock::ones() * 1E-6f, EaBlock::ones() * 1E-6f}, 32, stream);
    checkCudaErrors(cudaStreamSynchronize(stream));
    const Eigen::VectorXd ref = SchurVec2Eigen(x0.getParamConst());
    const Eigen::VectorXd solution = SchurVec2Eigen(x.getParamConst());
//    Eigen::MatrixXd cmp(ref.rows(), 3);
//    cmp << ref, solution, solution - ref;
//    std::cout << cmp << std::endl;
    EXPECT_TRUE(((solution - ref).array().abs() < 1E-3f).all());
}
template <typename Traits>
void PCGTest<Traits>::dumpProblem() {
#define USING_HESSIAN_TYPE(type) using type = typename rba::HessianBase<Traits>::type
    USING_HESSIAN_TYPE(EcBlock);
    USING_HESSIAN_TYPE(EaBlock);
    USING_HESSIAN_TYPE(EbBlock);
    USING_HESSIAN_TYPE(MBlock);
    USING_HESSIAN_TYPE(UBlock);
    USING_HESSIAN_TYPE(VBlock);
#undef USING_HESSIAN_TYPE
    std::ofstream("x0.txt", std::ios::trunc) << SchurVec2Eigen(x0.getParamConst());
    std::ofstream("A.txt", std::ios::trunc) << DeviceSchur2Eigen(schur);
    std::ofstream("x.txt", std::ios::trunc) << SchurVec2Eigen(x.getParamConst());
    std::ofstream("b.txt", std::ios::trunc) << SchurVec2Eigen(this->b);

    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(sizeC * EcBlock::rows() + sizeA*EaBlock::rows(), sizeC * EcBlock::rows() + sizeA*EaBlock::rows());
    if constexpr(MBlock::rows() != 0) {
        for (uint32_t i = 0; i < schur.diagM.size(); i++)
            P.block<MBlock::rows(), MBlock::cols()>(MBlock::rows() * i, MBlock::cols() * i) = toEigenMap(
                    schur.diagM[i].toKMat()).template cast<double>();
    }
    for (uint32_t i = 0; i < schur.diagU.size(); i++)
        P.block<UBlock::rows(), UBlock::cols()>(MBlock::rows() * schur.diagM.size() + UBlock::rows() * i, MBlock::cols() * schur.diagM.size() + UBlock::cols() * i) = toEigenMap(schur.diagU[i].toKMat()).template cast<double>();
    std::ofstream("P.txt", std::ios::trunc) << P;
}

#define RBA_DEFINE_PCGTest(r, data, TRAITS)\
    using BOOST_PP_CAT(PCGTest_, TRAITS) = PCGTest<rba::TRAITS>;\
    TEST_F(BOOST_PP_CAT(PCGTest_, TRAITS), MMV)\
    {\
        testMMV();\
    }\
    TEST_F(BOOST_PP_CAT(PCGTest_, TRAITS), ComputeR)\
    {\
        testComputeR();\
    }\
    TEST_F(BOOST_PP_CAT(PCGTest_, TRAITS), Solve)\
    {\
        testSolve();\
    }
BOOST_PP_SEQ_FOR_EACH(RBA_DEFINE_PCGTest, data, ALL_TRAITS)
#undef RBA_DEFINE_PCGTest

