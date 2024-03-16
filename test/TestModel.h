//
// Created by yao on 28/10/18.
//

#pragma once
#include "../utils_host.h"
#include "../kernel.h"
#include "../csr.h"
#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>
#include <unordered_map>
#include "../computeHessian.h"
#include "../computeSchur.h"
#include <queue>
#include "../containers.h"
#include "../GroupModel.h"

template <typename Traits, template <typename> class BaseHostModel>
class TestModelBase : public BaseHostModel<Traits> //rba::grouped::HostModel<rba::TraitsGrpF2C2D5Global>//rba::grouped::GroupModel
{
public:
    RBA_IMPORT_TRAITS(Traits);
protected:
    void SetUp();
    void computeHostJacobian();
    void computeHostHessian();
    void computeHostSchur();
    virtual void computeHostReference() = 0;

    void setUpModel(size_t nbPts, size_t nbIntri, size_t nbCaps);

protected:
    size_t nbPts = 128;
    size_t nbIntri = rba::isGroupModel<Traits>() ? 4 : 0;
    size_t nbCaps = 8;
//    std::default_random_engine rng = std::default_random_engine{0};
    std::mt19937 rng{0};
    std::uniform_real_distribution<float> dist = std::uniform_real_distribution<float>{-.5f, .5f};
    const lpf omega = 1.f;
    const lpf huber = INFINITY;
    rba::grouped::HostModel<Traits>& hostModel = *this;

	// reference
	std::vector<rba::Point<lpf>> refPointList;
	std::vector<CamIntr> refIntriList;
	std::vector<Capture> refCapList;

    //these are full jacobian/hessian/error assuming that all intrinsics/extrinsics/points are variable
    Eigen::Matrix<lpf, -1, -1> hostJacobian;
    Eigen::Matrix<lpf, -1, 1> hostError;
    Eigen::Matrix<epf, -1, -1> hostHessian;
    Eigen::Matrix<epf, -1, 1> hostHessianEpsilon;
    Eigen::Matrix<epf, -1, -1> hostSchur;
    Eigen::Matrix<epf, -1, 1> hostSchurEpsilon;

    cudaStream_t stream = nullptr;
};

#define RBA_TEST_IMPORT_TestModelBase_MEMBERS(Traits, ModelType)\
    using TestModelBase<Traits, ModelType>::nbPts;\
    using TestModelBase<Traits, ModelType>::nbIntri;\
    using TestModelBase<Traits, ModelType>::nbCaps;\
    using TestModelBase<Traits, ModelType>::rng;\
    using TestModelBase<Traits, ModelType>::omega;\
    using TestModelBase<Traits, ModelType>::huber;\
    using TestModelBase<Traits, ModelType>::hostModel;\
    using TestModelBase<Traits, ModelType>::hostJacobian;\
    using TestModelBase<Traits, ModelType>::hostError;\
    using TestModelBase<Traits, ModelType>::hostHessian;\
    using TestModelBase<Traits, ModelType>::hostHessianEpsilon;\
    using TestModelBase<Traits, ModelType>::hostSchur;\
    using TestModelBase<Traits, ModelType>::hostSchurEpsilon;\
    using TestModelBase<Traits, ModelType>::computeHostJacobian;\
    using TestModelBase<Traits, ModelType>::computeHostHessian;\
    using TestModelBase<Traits, ModelType>::computeHostSchur;\
    using TestModelBase<Traits, ModelType>::computeHostReference

template <typename Traits>
class TestModel : public ::testing::Test, public TestModelBase<Traits, rba::grouped::GroupModel>
{
public:
    RBA_IMPORT_TRAITS(Traits);
    void SetUp() override;
protected:
    std::unique_ptr<rba::grouped::DeviceModel<Traits>> devModel = std::make_unique<rba::grouped::DeviceModel<Traits>>();
    std::unique_ptr<rba::grouped::DeviceHessian<Traits>> devHessian = std::make_unique<rba::grouped::DeviceHessian<Traits>>();
    std::unique_ptr<rba::grouped::DeviceHessianPreCompData> devHessianPreComp = std::make_unique<rba::grouped::DeviceHessianPreCompData>();
    using VSymBlock = symkmat<lpf, rba::Point<lpf>::DoF>;
    std::unique_ptr<VSymBlock, CudaDeviceDeleter> devInvVBlocks;
    std::unique_ptr<rba::grouped::DeviceSchur<Traits>> devSchur = std::make_unique<rba::grouped::DeviceSchur<Traits>>();
    std::unique_ptr<rba::grouped::DeviceSchurPreCompData> devSchurPreComp = std::make_unique<rba::grouped::DeviceSchurPreCompData>();
};

#define RBA_TEST_IMPORT_TestModel_MEMBERS(Traits)\
    RBA_TEST_IMPORT_TestModelBase_MEMBERS(Traits, rba::grouped::GroupModel);\
    using TestModel<Traits>::computeHostHessian;\
    using TestModel<Traits>::computeHostJacobian;\
    using TestModel<Traits>::devModel;\
    using TestModel<Traits>::devHessian;\
    using TestModel<Traits>::devHessianPreComp;\
    using TestModel<Traits>::stream;\
    using TestModel<Traits>::devSchur;\
    using TestModel<Traits>::devSchurPreComp;\
    using TestModel<Traits>::devInvVBlocks;\
    using rba::grouped::GroupModel<Traits>::initializeOptimization;\
    using rba::grouped::GroupModel<Traits>::optimize;\
    using rba::grouped::GroupModel<Traits>::nbVarIntri;\
    using rba::grouped::GroupModel<Traits>::nbVarCaps;\
    using rba::grouped::GroupModel<Traits>::nbVarPoints

template <typename Intrinsics>
struct RandParamGen;

template <typename ftype, bool rolling, typename Func>
std::conditional_t<rolling, ftype, std::monostate> makeRandRollingCenter(const Func& genRand) {
	if constexpr (rolling) {
		return genRand() * 100.f + 500.f;
	}
	else {
		return {};
	}
}

template <typename ftype, bool rolling>
struct RandParamGen<rba::IntrinsicsF2C2D5<ftype, rolling>>{
    template <typename Func>
    static typename rba::IntrinsicsF2C2D5<ftype, rolling> make(const Func& genRand)
    {
        return {
                {genRand()*200+1000, genRand()*200+1000},
                {genRand()*200+1000, genRand()*200+1000},
                {genRand()*1E-2f, genRand()*1E-2f, genRand()*1E-3f, genRand()*1E-3f, genRand()*1E-3f},
				makeRandRollingCenter<ftype, rolling, Func>(genRand)
        };
    }
};

template <typename ftype, bool rolling>
struct RandParamGen<rba::IntrinsicsF2D5<ftype, rolling>>{
    template <typename Func>
    static typename rba::IntrinsicsF2D5<ftype, rolling> make(const Func& genRand)
    {
        return {
                {genRand()*200+1000, genRand()*200+1000},
                {genRand()*1E-2f, genRand()*1E-2f, genRand()*1E-3f, genRand()*1E-3f, genRand()*1E-3f},
				makeRandRollingCenter<ftype, rolling, Func>(genRand)
        };
    }
};

template <typename ftype, bool rolling>
struct RandParamGen<rba::IntrinsicsF1C2D5<ftype, rolling>>{
    template <typename Func>
    static typename rba::IntrinsicsF1C2D5<ftype, rolling> make(const Func& genRand)
    {
        return {
                genRand()*200+1000,
                {genRand()*1000, genRand()*1000},
                {genRand()*1E-2f, genRand()*1E-2f, genRand()*1E-3f, genRand()*1E-3f, genRand()*1E-3f},
				makeRandRollingCenter<ftype, rolling, Func>(genRand)
        };
    }
};

template <typename ftype, bool rolling>
struct RandParamGen<rba::IntrinsicsF1D5<ftype, rolling>>{
    template <typename Func>
    static typename rba::IntrinsicsF1D5<ftype, rolling> make(const Func& genRand)
    {
        return {
                genRand()*200+1000,
                {genRand()*1E-2f, genRand()*1E-2f, genRand()*1E-3f, genRand()*1E-3f, genRand()*1E-3f},
				makeRandRollingCenter<ftype, rolling, Func>(genRand)
        };
    }
};

template <typename ftype, bool rolling>
struct RandParamGen<rba::IntrinsicsF1D2<ftype, rolling>>{
    template <typename Func>
    static typename rba::IntrinsicsF1D2<ftype, rolling> make(const Func& genRand)
    {
        return {
                genRand()*200+1000,
                {genRand()*1E-2f, genRand()*1E-2f},
				makeRandRollingCenter<ftype, rolling, Func>(genRand)
        };
    }
};

template <typename ftype, bool rolling>
struct RandParamGen<rba::IntrinsicsF1<ftype, rolling>>{
    template <typename Func>
    static typename rba::IntrinsicsF1<ftype, rolling> make(const Func& genRand)
    {
        return {
			genRand()*200+1000,
			{},
			makeRandRollingCenter<ftype, rolling, Func>(genRand)
		};
    }
};


template <typename ftype, bool rolling>
struct RandParamGen<rba::IntrinsicsF2<ftype, rolling>>{
    template <typename Func>
    static typename rba::IntrinsicsF2<ftype, rolling> make(const Func& genRand)
    {
        return {
			{genRand()*200+1000, genRand()*200+1000},
			{},
			makeRandRollingCenter<ftype, rolling, Func>(genRand)
        };
    }
};


template <typename ftype, bool rolling>
struct RandParamGen<rba::IntrinsicsF2C2<ftype, rolling>>{
    template <typename Func>
    static typename rba::IntrinsicsF2C2<ftype, rolling> make(const Func& genRand)
    {
        return {
			{genRand()*200+1000, genRand()*200+1000},
			{genRand()*1000, genRand()*1000},
			{},
			makeRandRollingCenter<ftype, rolling, Func>(genRand)
        };
    }
};

template <typename ftype>
struct RandParamGen<rba::Pose<ftype>>{
    template <typename Func>
    static typename rba::Pose<ftype> make(const Func& genRand)
    {
        auto q = (Eigen::Matrix<ftype, 4, 1>{} << genRand(), genRand(), genRand(), genRand()).finished().normalized();
        return rba::Pose<ftype>{{q[0], q[1], q[2], q[3]}, {genRand(), genRand(), genRand()}};
    }
};
