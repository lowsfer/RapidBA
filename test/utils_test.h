//
// Created by yao on 28/10/18.
//

#pragma once
#include "../kmat.h"
#include "../GroupModelTypes.h"
#include <eigen3/Eigen/Dense>
#include <type_traits>
#include <gtest/gtest.h>
#include <boost/format.hpp>

template <typename T, typename Dummy = typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value, void>::type>
T absOrRelativeDiff(T a, T b){
    static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "fatal error");
    const T diff = std::abs(a-b);
    return std::min(diff, diff / std::max(std::abs(a), std::abs(b)));
}

#define CHECK_KMAT_CLOSE(a, b, threshold, seed) \
    do{\
        for (unsigned i = 0; i < (a).rows(); i++) \
            for (unsigned j = 0; j < (a).cols(); j++) \
                EXPECT_LE(absOrRelativeDiff((a)(i,j), (b)(i, j)), (threshold)) << boost::format("Difference at (%d, %d): %f, %f (seed = %u)") % i % j % (a)(i,j) % (b)(i,j) % (seed);\
    }while(false)

template <typename Traits>
Eigen::MatrixXd DeviceSchur2Eigen(const rba::grouped::DeviceSchur<Traits>& schur);
template <typename Traits>
Eigen::VectorXd SchurVec2Eigen(const rba::pcg::SchurVec<Traits, true>& vec);
template <typename Traits>
Eigen::VectorXd DeviceSchurEpsilon2Eigen(const rba::grouped::DeviceSchur<Traits>& schur);
template <typename Traits>
std::pair<Eigen::MatrixXd, Eigen::VectorXd> DeviceHessianAndEpsilon2Eigen(const rba::grouped::DeviceHessian<Traits>& hessian);
template <typename Traits>
Eigen::MatrixXd DeviceHessian2Eigen(const rba::grouped::DeviceHessian<Traits>& hessian);
template <typename Traits>
Eigen::VectorXd HessianVec2Eigen(const rba::HessianVec<Traits, true>& vec);
template <typename Traits>
Eigen::VectorXd DeviceHessianEpsilon2Eigen(const rba::grouped::DeviceHessian<Traits>& hessian);

template <typename SymKMatType, typename Func>
SymKMatType genPositiveDefiniteSymKMat(Func&& rng, float diagFactor = 0.25f) {
    using EMat = Eigen::Matrix<typename SymKMatType::ValType, SymKMatType::rows(), SymKMatType::cols()>;
    EMat m = EMat::NullaryExpr([&rng]() { return typename SymKMatType::ValType(rng()); });

    kmat<typename SymKMatType::ValType, SymKMatType::rows(), SymKMatType::cols()> result;
    toEigenMap(result) = m * m.transpose() + EMat::Identity() * diagFactor;
    return result;
}