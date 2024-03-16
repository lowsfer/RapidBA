//
// Created by yao on 5/12/18.
//

#include "utils_test.h"
#include <boost/preprocessor/seq/for_each.hpp>

using rba::TraitsGrpF2C2D5Global;
using rba::TraitsGrpF1D2Global;

template <typename Traits>
Eigen::MatrixXd DeviceSchur2Eigen(const rba::grouped::DeviceSchur<Traits>& schur){
    using MBlock = typename rba::HessianBase<Traits>::MBlock;
    using UBlock = typename rba::HessianBase<Traits>::UBlock;
    Eigen::MatrixXd denseSchur = Eigen::Matrix<double, -1, -1>::Zero(MBlock::rows() * schur.diagM.size() + UBlock::rows() * schur.diagU.size(),
                                                                     MBlock::cols() * schur.diagM.size() + UBlock::rows() * schur.diagU.size());

    using ScalarType = typename std::remove_reference<decltype(denseSchur(0,0))>::type;

    const auto upperM = schur.upperM.toEigen(schur.diagM.size()).template cast<ScalarType>().eval();
    denseSchur.block(0, 0, MBlock::rows() * schur.diagM.size(), MBlock::cols() * schur.diagM.size()) = upperM + upperM.transpose();
    for (uint32_t i = 0; i < schur.diagM.size(); i++)
        denseSchur.block<MBlock::rows(), MBlock::cols()>(MBlock::rows() * i, MBlock::cols() * i) = toEigenMap(schur.diagM[i].toKMat()).template cast<ScalarType>();

    const auto upperU = schur.upperU.toEigen(schur.diagU.size()).template cast<ScalarType>().eval();
    denseSchur.block(MBlock::rows() * schur.diagM.size(), MBlock::cols() * schur.diagM.size(), UBlock::rows() * schur.diagU.size(), UBlock::cols() * schur.diagU.size()) = upperU + upperU.transpose();
    for (uint32_t i = 0; i < schur.diagU.size(); i++)
        denseSchur.block<UBlock::rows(), UBlock::cols()>(MBlock::rows() * schur.diagM.size() + UBlock::rows() * i, MBlock::cols() * schur.diagM.size() + UBlock::cols() * i) = toEigenMap(schur.diagU[i].toKMat()).template cast<ScalarType>();

    const auto Q = schur.Q.toEigen(schur.diagU.size()).template cast<ScalarType>().eval();
    denseSchur.block(0, MBlock::cols() * schur.diagM.size(), MBlock::rows() * schur.diagM.size(), UBlock::cols() * schur.diagU.size()) = Q;
    denseSchur.transpose().block(0, MBlock::cols() * schur.diagM.size(), MBlock::rows() * schur.diagM.size(), UBlock::cols() * schur.diagU.size()) = Q;

    assert((denseSchur.transpose() - denseSchur).norm() < 1E-4f);
    return denseSchur;
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template Eigen::MatrixXd DeviceSchur2Eigen(const rba::grouped::DeviceSchur<rba::TRAITS>& schur);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template <typename Traits>
Eigen::VectorXd SchurVec2Eigen(const rba::pcg::SchurVec<Traits, true>& vec){
    using EcBlock = typename rba::HessianBase<Traits>::EcBlock;
    using EaBlock = typename rba::HessianBase<Traits>::EaBlock;
    Eigen::VectorXd schurEpsilon = Eigen::VectorXd::Zero(EcBlock::rows() * vec.nbCBlocks + EaBlock::rows() * vec.nbABlocks);
    using ScalarType = typename std::remove_reference<decltype(schurEpsilon(0,0))>::type;
    for (uint32_t i = 0; i < vec.nbCBlocks; i++)
        schurEpsilon.block<EcBlock::rows(), 1>(EcBlock::rows()*i, 0) = toEigenMap(vec.c[i]).template cast<ScalarType>();
    for (uint32_t i = 0; i < vec.nbABlocks; i++)
        schurEpsilon.block<EaBlock::rows(), 1>(EcBlock::rows()*vec.nbCBlocks + EaBlock::rows()*i, 0) = toEigenMap(vec.a[i]).template cast<ScalarType>();
    return schurEpsilon;
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template Eigen::VectorXd SchurVec2Eigen(const rba::pcg::SchurVec<rba::TRAITS, true>& vec);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template <typename Traits>
Eigen::VectorXd DeviceSchurEpsilon2Eigen(const rba::grouped::DeviceSchur<Traits>& schur){
    const rba::pcg::SchurVec<Traits, true> schurVec{
        schur.Ec.data(),
       cast32u(schur.Ec.size()),
        schur.Ea.data(),
       cast32u(schur.Ea.size())
    };
    return SchurVec2Eigen(schurVec);
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template Eigen::VectorXd DeviceSchurEpsilon2Eigen(const rba::grouped::DeviceSchur<rba::TRAITS>& schur);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template <typename Traits>
std::pair<Eigen::MatrixXd, Eigen::VectorXd> DeviceHessianAndEpsilon2Eigen(const rba::grouped::DeviceHessian<Traits>& hessian)
{
    RBA_IMPORT_TRAITS(Traits);
    constexpr size_t blockSize[3] = {CamIntr::DoF, Capture::DoF, rba::Point<lpf>::DoF};
    const size_t offsets[3] = {0u, blockSize[0] * hessian.M.size(), blockSize[0] * hessian.M.size() + blockSize[1] * hessian.U.size()};
    const size_t length[3] = {blockSize[0] * hessian.M.size(), blockSize[1] * hessian.U.size(), blockSize[2] * hessian.V.size()};
    Eigen::MatrixXd cudaHessian = Eigen::MatrixXd::Zero(offsets[2] + length[2], offsets[2] + length[2]);
    Eigen::VectorXd cudaEpsilon = Eigen::VectorXd::Zero(offsets[2] + length[2]);
    auto hostU = cudaHessian.block(offsets[1], offsets[1], length[1], length[1]);
    auto hostV = cudaHessian.block(offsets[2], offsets[2], length[2], length[2]);
    auto hostW = cudaHessian.block(offsets[1], offsets[2], length[1], length[2]);

    for (unsigned i = 0; i < hessian.U.size(); i++) {
        hostU.block<blockSize[1], blockSize[1]>(blockSize[1] * i, blockSize[1] * i) = toEigenMap(
                hessian.U[i].toKMat()).template cast<epf>();
        cudaEpsilon.block<blockSize[1],1>(offsets[1]+blockSize[1] * i, 0) = toEigenMap(hessian.Ea[i]).template cast<epf>();
    }
    for (unsigned i = 0; i < hessian.V.size(); i++) {
        hostV.block<blockSize[2], blockSize[2]>(blockSize[2] * i, blockSize[2] * i) = toEigenMap(
                hessian.V[i].toKMat()).template cast<epf>();
        cudaEpsilon.block<blockSize[2],1>(offsets[2]+blockSize[2] * i, 0) = toEigenMap(hessian.Eb[i]).template cast<epf>();
    }
    for (unsigned i = 0; i < hessian.W.nbRows; i++){
        for (unsigned j = hessian.W.rows[i]; j < hessian.W.rows[i+1]; j++){
            hostW.block<blockSize[1], blockSize[2]>(blockSize[1]*i, blockSize[2]*hessian.W.idxCol[j]) = toEigenMap(hessian.W.data[j]).template cast<epf>();
        }
    }
    cudaHessian.block(offsets[2], offsets[1], length[2], length[1]) = hostW.transpose();
    if constexpr(CamIntr::DoF != 0) {
        auto hostM = cudaHessian.block(offsets[0], offsets[0], length[0], length[0]);
        auto hostQ = cudaHessian.block(offsets[0], offsets[1], length[0], length[1]);
        auto hostS = cudaHessian.block(offsets[0], offsets[2], length[0], length[2]);
        for (unsigned i = 0; i < hessian.M.size(); i++) {
            hostM.block<blockSize[0], blockSize[0]>(blockSize[0] * i, blockSize[0] * i) = toEigenMap(
                    hessian.M[i].toKMat()).template cast<epf>();
            cudaEpsilon.block<blockSize[0],1>(offsets[0]+blockSize[0] * i, 0) = toEigenMap(hessian.Ec[i]).template cast<epf>();
        }
        for (unsigned i = 0; i < hessian.Q.blocks.size(); i++) {
            if (hessian.Q.row[i] != rba::ModelBase<Traits>::varIdxFixed)
                hostQ.block<blockSize[0], blockSize[1]>(blockSize[0] * hessian.Q.row[i], blockSize[1] * i)
                        = toEigenMap(hessian.Q.blocks[i]).template cast<epf>();
        }
        for (unsigned i = 0; i < hessian.S.nbRows; i++){
            for (unsigned j = hessian.S.rows[i]; j < hessian.S.rows[i+1]; j++){
                hostS.block<blockSize[0], blockSize[2]>(blockSize[0]*i, blockSize[2]*hessian.S.idxCol[j]) = toEigenMap(hessian.S.data[j]).template cast<epf>();
            }
        }
        cudaHessian.block(offsets[1], offsets[0], length[1], length[0]) = hostQ.transpose();
        cudaHessian.block(offsets[2], offsets[0], length[2], length[0]) = hostS.transpose();
    }
    return {cudaHessian, cudaEpsilon};
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template std::pair<Eigen::MatrixXd, Eigen::VectorXd> DeviceHessianAndEpsilon2Eigen(const rba::grouped::DeviceHessian<rba::TRAITS>& hessian);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template <typename Traits>
Eigen::MatrixXd DeviceHessian2Eigen(const rba::grouped::DeviceHessian<Traits>& hessian)
{
    return DeviceHessianAndEpsilon2Eigen(hessian).first;
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template Eigen::MatrixXd DeviceHessian2Eigen(const rba::grouped::DeviceHessian<rba::TRAITS>& hessian);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template <typename Traits>
Eigen::VectorXd HessianVec2Eigen(const rba::HessianVec<Traits, true>& vec)
{
    const auto ca = SchurVec2Eigen<Traits>({vec.c, vec.nbCBlocks, vec.a, vec.nbABlocks});
    using EbBlock = typename rba::HessianBase<Traits>::EbBlock;
    Eigen::VectorXd b(EbBlock::rows() * vec.nbBBlocks);
    for (uint32_t i = 0; i < vec.nbBBlocks; i++){
        b.block<EbBlock::rows(), 1>(EbBlock::rows() * i, 0) = toEigenMap(vec.b[i]).template cast<double>();
    }
    Eigen::VectorXd cab(ca.rows() + b.rows());
    cab << ca, b;
    return cab;
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template Eigen::VectorXd HessianVec2Eigen(const rba::HessianVec<rba::TRAITS, true>& vec);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES

template <typename Traits>
Eigen::VectorXd DeviceHessianEpsilon2Eigen(const rba::grouped::DeviceHessian<Traits>& hessian)
{
    return DeviceHessianAndEpsilon2Eigen(hessian).second;
}
#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template Eigen::VectorXd DeviceHessianEpsilon2Eigen(const rba::grouped::DeviceHessian<rba::TRAITS>& hessian);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES
