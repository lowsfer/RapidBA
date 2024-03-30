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

#include "RapidBA.h"
#include "GroupModel.h"
#include <iostream>
#include "kernel.h"
#include "DiscreteModel.h"
#include "UniversalModel.h"

namespace rba {

ShutterType IModel::getShutterType() const {return ShutterType::kGlobal;}
bool IModel::hasRollingShutter() const {return isRolling(getShutterType());}

//template <typename CaptureParamsType>
//bool IDiscreteModel<CaptureParamsType>::isGrouped() const {return false;}

bool IGroupModelBase::isGrouped() const {return true;}

template <ShutterType shutterType>
IGroupModelF2C2D5<shutterType>* createGroupModelF2C2D5() {
    return new grouped::GroupModel<TraitsGrpF2C2D5Template<shutterType>>();
}

template <ShutterType shutterType>
IGroupModelF2D5<shutterType>* createGroupModelF2D5() {
    return new grouped::GroupModel<TraitsGrpF2D5Template<shutterType>>();
}

template <ShutterType shutterType>
IGroupModelF1C2D5<shutterType>* createGroupModelF1C2D5() {
    return new grouped::GroupModel<TraitsGrpF1C2D5Template<shutterType>>();
}

template <ShutterType shutterType>
IGroupModelF1D5<shutterType>* createGroupModelF1D5() {
    return new grouped::GroupModel<TraitsGrpF1D5Template<shutterType>>();
}

template <ShutterType shutterType>
IGroupModelF1D2<shutterType>* createGroupModelF1D2() {
    return new grouped::GroupModel<TraitsGrpF1D2Template<shutterType>>();
}

template <ShutterType shutterType>
IGroupModelF1<shutterType>* createGroupModelF1() {
    return new grouped::GroupModel<TraitsGrpF1Template<shutterType>>();
}

template <ShutterType shutterType>
IGroupModelF2C2<shutterType>* createGroupModelF2C2() {
    return new grouped::GroupModel<TraitsGrpF2C2Template<shutterType>>();
}

template <ShutterType shutterType>
IGroupModelF2<shutterType>* createGroupModelF2() {
    return new grouped::GroupModel<TraitsGrpF2Template<shutterType>>();
}

#define INSTANTIATE_CREATE_GROUP_MODEL(shutter) \
	template IGroupModelF2C2D5<shutter>* createGroupModelF2C2D5(); \
	template IGroupModelF2D5<shutter>* createGroupModelF2D5(); \
	template IGroupModelF1C2D5<shutter>* createGroupModelF1C2D5(); \
	template IGroupModelF1D5<shutter>* createGroupModelF1D5(); \
	template IGroupModelF1D2<shutter>* createGroupModelF1D2(); \
	template IGroupModelF1<shutter>* createGroupModelF1(); \
	template IGroupModelF2C2<shutter>* createGroupModelF2C2(); \
	template IGroupModelF2<shutter>* createGroupModelF2();
INSTANTIATE_CREATE_GROUP_MODEL(ShutterType::kGlobal)
INSTANTIATE_CREATE_GROUP_MODEL(ShutterType::kRollingFixedVelocity)
INSTANTIATE_CREATE_GROUP_MODEL(ShutterType::kRolling1D)
INSTANTIATE_CREATE_GROUP_MODEL(ShutterType::kRolling1DLoc)
INSTANTIATE_CREATE_GROUP_MODEL(ShutterType::kRolling3D)
#undef INSTANTIATE_CREATE_GROUP_MODEL

IUniversalModel* createUniversalModel(bool useGroupModel, IModel::IntriType intriType, ShutterType shutterType) {
    return new UniversalModel(useGroupModel, intriType, shutterType);
}

IDiscreteModelF1D2* createDiscreteModelF1D2() {
    return new DiscreteModel<TraitsDisF1D2>();
}

IDiscreteModelF1* createDiscreteModelF1() {
    return new DiscreteModel<TraitsDisF1>();
}

}
