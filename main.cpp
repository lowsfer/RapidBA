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
// Created by yao on 28/11/18.
//

#include <iostream>
#include "RapidBA.h"
#include <memory>
#include <fstream>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>
#include <chrono>
#include <boost/format.hpp>
#include <condition_variable>
#include <mutex>
#include <chrono>
#include <thread>

#pragma GCC push_options
#pragma GCC optimize ("O0")

using boost::format;
using boost::lexical_cast;

const float huber = 1.f;

template <typename Src>
float toFloat(const Src& src)
{
    return boost::lexical_cast<float>(src);
}

void createGroupModelFromFile(rba::IUniversalModel* model, const std::string &filename) {
    model->clear();

    std::ifstream fin;
    fin.exceptions(std::ios::badbit | std::ios::failbit | std::ios::eofbit);
    fin.open(filename);

    std::string line;
    std::getline(fin, line);
    boost::escaped_list_separator<char> sep("\\", " \t", "\"\'");
    boost::tokenizer<boost::escaped_list_separator<char>> tokens(line, sep);
    std::vector<std::string> token_list;
    std::copy(tokens.begin(), tokens.end(), std::back_inserter(token_list));
    assert(token_list[0] == "NVM_V3");
    assert(token_list[1] == "FixedK");
    const rba::IUniversalModel::CamParamType cam{
            {toFloat(token_list.at(2)), toFloat(token_list.at(4))},
            // We are using openCV-style optical center in NVM files
            {toFloat(token_list.at(3)) + 0.5f, toFloat(token_list.at(5)) + 0.5f},
            0.f, 0.f, 0.f, 0.f, 0.f,
			toFloat(token_list.at(5))
    };
    std::vector<rba::IdxCam> cameras;
    const rba::IdxCam idxCam = model->addCamera(cam, false);
    cameras.emplace_back(idxCam);
    std::getline(fin, line);
    const int num_cameras = lexical_cast<int>(line);
    std::vector<rba::IdxCap> captures;
    std::vector<std::string> images;
    for(int i = 0; i < num_cameras; i++)
    {
        rba::IUniversalModel::CapParamType view{};
        std::getline(fin, line);
        tokens.assign(line, sep);
        auto iter = tokens.begin();
        images.emplace_back(*iter++);
        const float f = lexical_cast<float>(*iter++);
        (void)(f);
        Eigen::Quaternionf q;
        q.w() = lexical_cast<float>(*iter++);
        q.x() = lexical_cast<float>(*iter++);
        q.y() = lexical_cast<float>(*iter++);
        q.z() = lexical_cast<float>(*iter++);
        view.q = {q.x(), q.y(), q.z(), q.w()};
        Eigen::Vector3d C;
        C[0] = lexical_cast<double>(*iter++);
        C[1] = lexical_cast<double>(*iter++);
        C[2] = lexical_cast<double>(*iter++);
        Eigen::Vector3d::Map(&view.c.x) = C;
		view.velocity = {0.f, -1E-8f, 0.f};
        const rba::IdxCap idxCap = model->addCapture(idxCam, view);
        if (i == 0){
            model->setCaptureFixed(idxCap, true);
        }
        captures.emplace_back(idxCap);
    }
    std::getline(fin, line);
    const unsigned num_points = lexical_cast<unsigned>(line);
    std::vector<rba::IdxPt> points;
    size_t num_observations = 0u;
    for(unsigned i = 0; i < num_points; i++)
    {
        double3 point{};
        std::getline(fin, line);
        tokens.assign(line, sep);
        auto iter = tokens.begin();
        {
            const auto x = lexical_cast<double>(*iter++);
            const auto y = lexical_cast<double>(*iter++);
            const auto z = lexical_cast<double>(*iter++);
            point = {x, y, z};
        }
        const rba::IdxPt idxPt = model->addPoint(point);
        points.emplace_back(idxPt);
        //skip color
        iter++; iter++; iter++;
        const unsigned num_observations_this = lexical_cast<unsigned>(*iter++);
        for(unsigned j = 0; j < num_observations_this; j++)
        {
            const rba::IdxCap idxCap = lexical_cast<float>(*iter++);
            const auto idxKpoint = lexical_cast<float>(*iter++);
            (void)(idxKpoint);
            const float2 pt2d = {lexical_cast<float>(*iter++), lexical_cast<float>(*iter++)};
            model->addObservation(idxCap, idxPt, pt2d, 1.f, huber);
        }
        num_observations += num_observations_this;
    }
    std::cout << format("Model has %u cameras, %u captures, %u points and %u observations")
        % cameras.size() % captures.size() % points.size() % num_observations << std::endl;
}

std::unique_ptr<rba::IUniversalModel> createGroupModelFromFile(const std::string &filename) {
    std::unique_ptr<rba::IUniversalModel> model{rba::createUniversalModel(true, rba::IModel::IntriType::kF2C2D5, rba::ShutterType::kRolling1DLoc)};
    createGroupModelFromFile(model.get(), filename);
    return model;
}

void createDiscreteModelFromFile(rba::IDiscreteModelF1D2* model, const std::string &filename) {
    model->clear();
    constexpr float fScale = 1.f;
    std::ifstream fin;
    fin.exceptions(std::ios::badbit | std::ios::failbit | std::ios::eofbit);
    fin.open(filename);

    struct ObData{
        uint32_t idxCap;
        uint32_t idxPt;
        float2 proj;
    };
    std::vector<ObData> observations;
    uint32_t nbCap, nbPt, nbOb;
    fin >> nbCap >> nbPt >> nbOb;
    observations.reserve(nbOb);
    for (uint32_t i = 0; i < nbOb; i++){
        ObData ob{};
        fin >> ob.idxCap >> ob.idxPt >> ob.proj.x >> ob.proj.y;
        ob.proj.x = -fScale * ob.proj.x;
        ob.proj.y = -fScale * ob.proj.y;
        observations.emplace_back(ob);
    }
    std::vector<rba::IdxCap> captures;
    captures.reserve(nbCap);

    for (uint32_t i = 0; i < nbCap; i++) {
        Eigen::Vector3f rvec;
        rba::discrete::CapParamTypeF1D2 cap{};
        fin >> rvec[0] >> rvec[1] >> rvec[2];
        Eigen::Vector3d t;
        fin >> t[0] >> t[1] >> t[2];
        fin >> cap.intrinsics.f >> cap.intrinsics.k1 >> cap.intrinsics.k2;
        cap.intrinsics.f *= fScale;
        const float theta = rvec.norm();
        const float cosHalfTheta = std::cos(theta / 2);
        const float sinHalfTheta = std::sin(theta / 2);
        cap.pose.q.w = cosHalfTheta;
        Eigen::Vector3f::Map(&cap.pose.q.x) = sinHalfTheta / theta * rvec;
        const auto& q = cap.pose.q;
        Eigen::Vector3d::Map(&cap.pose.c.x) = Eigen::Quaternion<double>(q.w, q.x, q.y, q.z).conjugate() * (-t);
        const auto idxCap = model->addCapture(cap);
        captures.push_back(idxCap);
    }
    std::vector<rba::IdxPt> points;
    points.reserve(nbPt);
    for (uint32_t i = 0; i < nbPt; i++) {
        double3 pt;
        fin >> pt.x >> pt.y >> pt.z;
        const auto idxPt = model->addPoint(pt);
        points.push_back(idxPt);
    }
    for (const auto& ob : observations) {
        model->addObservation(ob.idxCap, ob.idxPt, ob.proj, 1.f, huber * fScale);
    }
    std::cout << format("Model has %u captures, %u points and %u observations")
                 % captures.size() % points.size() % nbOb << std::endl;
}


std::unique_ptr<rba::IDiscreteModelF1D2> createDiscreteModelFromFile(const std::string &filename) {
    std::unique_ptr<rba::IDiscreteModelF1D2> model{rba::createDiscreteModelF1D2()};
    createDiscreteModelFromFile(model.get(), filename);
    return model;
}


class WatchDog
{
public:
    WatchDog(const float maxSec) :mMaxSeconds{maxSec}, mThrd{&WatchDog::killer, this} {}
    WatchDog(const WatchDog&) = delete;
    WatchDog(WatchDog&&) = delete;
    WatchDog operator=(const WatchDog&) = delete;
    WatchDog operator=(WatchDog&&) = delete;
    ~WatchDog() {
        {
            std::lock_guard<std::mutex> lk{mLock};
            mFinished = true;
        }
        mCVar.notify_all();
        mThrd.join();
    }
private:
    void killer() const {
        std::unique_lock<std::mutex> lk{mLock};
        mCVar.wait_for(lk, std::chrono::duration<float>{mMaxSeconds}, [this](){ return mFinished; });
        if (!mFinished) {
            fprintf(stderr, "Timed out. Terminating process ...\n");
            std::terminate();
        }
    }
    mutable std::condition_variable mCVar;
    mutable std::mutex mLock;
    bool mFinished {false};
    float mMaxSeconds;
    std::thread mThrd;
};

int main(int argc, const char* argv[])
{
    WatchDog watchDog{600};
    const int maxIters = (argc >= 2 ? lexical_cast<int>(argv[1]) : 100);
    const float damp = (argc >= 3 ? lexical_cast<float>(argv[2]): 1E-2f);
#if 1
    const std::unique_ptr<rba::IUniversalModel> model = createGroupModelFromFile(
//            "/home/yao/projects/mapper2d/data/golf_small/output/cloud_0.nvm"
//    "/home/yao/projects/RapidBA/data/cloud_noBA0.nvm"
"/home/yao/projects/rapidsfm2/build/qtcreator-release/cloud_0.nvm"
//        "/home/yao/projects/RapidBA/data/cloud_noBA0_sift.nvm"
//        "/home/yao/projects/RapidBA/data/cloud_0_BA1.nvm"
// "/home/yao/projects/3D/data/airport/undistorted/output/cloud.nvm"
            );
#if 0
    // test model->clear()
    model->setVerbose(true);
    model->filterModel();
    model->initializeOptimization();
    model->setInitDamp(damp);
    model->optimize(maxIters);
    model->clear();
    createGroupModelFromFile(model.get(), "/home/yao/projects/RapidBA/data/cloud_0_BA1.nvm");
#endif
#else
    const std::unique_ptr<rba::IDiscreteModelF1D2> model = createDiscreteModelFromFile(
//            "/home/yao/projects/RapidBA/data/BAL/Dubrovnik/problem-356-226730-pre.txt"
            "/home/yao/projects/RapidBA/data/BAL/Venice/problem-1778-993923-pre.txt"
//            "/home/yao/projects/RapidBA/data/BAL/Trafalgar/problem-21-11315-pre.txt"
//            "/home/yao/projects/RapidBA/data/BAL/Trafalgar/problem-257-65132-pre.txt"
//            "/home/yao/projects/RapidBA/data/PBA/venice.txt"
            );
#endif
    model->setVerbose(true);
    model->filterModel();
    std::chrono::time_point<std::chrono::steady_clock> timePoints[3];
    timePoints[0] = std::chrono::steady_clock::now();
    model->initializeOptimization();
    model->setInitDamp(damp);
    timePoints[1] = std::chrono::steady_clock::now();
    model->optimize(maxIters);
    timePoints[2] = std::chrono::steady_clock::now();
    std::cout << "Time cost:" << std::endl;
    std::cout << format("  Initialization: %f ms.") % std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(timePoints[1] - timePoints[0]).count() << std::endl;
    std::cout << format("  Optimization: %f ms.") % std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(timePoints[2] - timePoints[1]).count() << std::endl;

	for (uint32_t i = 0; i < 642; i++) {
		const auto p = model->getCaptureParams(i);
		printf("#%3u: {%f, %f, %f} * 1E-4\n", i, p.velocity.x * 1E4f, p.velocity.y * 1E4f, p.velocity.z * 1E4f);
	}
    return 0;
}
#pragma GCC pop_options
