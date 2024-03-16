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