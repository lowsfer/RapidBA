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
// Created by yao on 9/6/19.
//

#pragma once

namespace rba{
namespace solver{
template <typename Traits> class HessianSolver;
template <typename Traits> class SchurSolver;
} // solver

namespace grouped {
template<typename Traits> struct HostModel;
template <typename Traits> struct DevHessianVec;
template <typename Traits> struct DeviceModelVarBackup;
template <typename Traits> struct DeviceModel;
struct DeviceHessianPreCompData;
template <typename Traits> struct DeviceHessian;
template <typename Traits> struct DevHessianVec;
struct DeviceSchurPreCompData;
template <typename Traits> struct DeviceSchur;
template <typename Traits> struct DevSchurVec;
template <typename Traits> struct DeviceModelVarBackup;
//@fixme: change namespace?
template <typename Traits> class BundleSolver;
template <typename Traits> class GroupModel;
}// grouped
}// rba