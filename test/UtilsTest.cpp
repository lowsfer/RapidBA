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
// Created by yao on 10/11/18.
//

#include <gtest/gtest.h>
#include "../utils_host.h"
#include "../kmat.h"
#include <eigen3/Eigen/Dense>

TEST(MergeUniqueTest, random)
{
    std::default_random_engine rng(0u);
    auto testOnce = [&](size_t nbRanges)->void{
        using DataType = unsigned;
        std::uniform_int_distribution<uint32_t> distRangeLen(0u, uint32_t(rng()%32));
        std::vector<uint32_t> ranges(nbRanges+1);
        ranges[0] = 0;
        for (size_t n = 0; n < nbRanges; n++){
            ranges[n+1] = ranges[n] + distRangeLen(rng);
        }
        std::uniform_int_distribution<DataType> distData(0, DataType(rng()%32));
        std::vector<DataType> srcData(ranges[nbRanges]);
        std::generate(srcData.begin(), srcData.end(), [&](){return distData(rng);});
        for (size_t n = 0; n < nbRanges; n++){
            std::sort(&srcData[ranges[n]], &srcData[ranges[n+1]]);
        }

        std::vector<DataType> dst;
        mergeUnique(dst, &srcData[0], ranges);

        std::vector<DataType> ref;
        std::vector<DataType> buffer;
        for (size_t n = 0; n < nbRanges; n++){
            swap(ref, buffer);
            ref.clear();
            std::merge(buffer.begin(), buffer.end(), srcData.begin()+ranges[n], srcData.begin()+ranges[n+1], std::back_inserter(ref));
            ref.erase(std::unique(ref.begin(), ref.end()), ref.end());
        }

        ASSERT_EQ(dst.size(), ref.size());
        for (size_t i = 0; i < dst.size(); i++){
            ASSERT_EQ(dst[i], ref[i]);
        }
    };
    for (const size_t nbRanges : {0u, 1u, 2u, 4u, 1024u}){
        testOnce(nbRanges);
    }
}

TEST(KMatTest, LLT)
{
    constexpr int dims = 5;
    const Eigen::Matrix<float, dims, dims> mat = [&]() {
        Eigen::Matrix<float, dims, dims*2> tmp = Eigen::Matrix<float, dims, dims*2>::Random();
        return (tmp * tmp.transpose() + Eigen::Matrix<float, dims, dims>::Identity()).eval();
    }();
    kmat<float, dims, dims> kMat;
    toEigenMap(kMat) = mat;
    const Eigen::Matrix<float, dims, dims> L = toEigenMap(cholesky(symkmat<float, dims>(kMat)));
    Eigen::Matrix<float, dims, dims> refL;
    mat.llt().matrixL().evalTo(refL);
    EXPECT_LE((refL - L).norm(), 1E-4);
}

TEST(KMatTest, LLTInv)
{
    constexpr int dims = 5;
    const Eigen::Matrix<float, dims, dims> mat = [&]() {
        Eigen::Matrix<float, dims, dims*2> tmp = Eigen::Matrix<float, dims, dims*2>::Random();
        return (tmp * tmp.transpose() + Eigen::Matrix<float, dims, dims>::Identity()).eval();
    }();
    kmat<float, dims, dims> kMat;
    toEigenMap(kMat) = mat;
    const Eigen::Matrix<float, dims, dims> invKMat = toEigenMap(inverseByCholesky(symkmat<float, dims>(kMat)).toKMat());
    EXPECT_LE((mat.inverse() - invKMat).norm(), 1E-4);
}

