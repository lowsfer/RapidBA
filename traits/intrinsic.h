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
// Created by yao on 9/4/19.
//

#pragma once
#include "../utils_general.h"
#include <type_traits>
#include <variant>
namespace rba{

//********************Distortion parameters**************************

//using ftype = FPTraitsSSD::lpf; template <uint32_t nbParams>
template <typename ftype, uint32_t nbParams>
struct DistortParams{
    DistortParams() = default;
    __device__ __host__ __forceinline__
    DistortParams(std::initializer_list<ftype>&& init) :params{std::move(init)}{}
    static_assert(nbParams <= 5, "Too many parameters");
    constexpr static uint32_t DoF = nbParams;
    // k1, k2, p1, p2, k3
    kmat<ftype, DoF> params;
#define DEFINE_ACCESSOR(name, index) \
    template <bool enabler = true, std::enable_if_t<enabler&&(DoF>index), int> = 0> \
    __device__ __host__ __forceinline__ ftype name() const { return params[index];} \
    template <bool enabler = true, std::enable_if_t<enabler&&(DoF<=index), int> = 0> \
    __device__ __host__ __forceinline__ sym::Zero name() const { return sym::Zero{};} \
    template <bool enabler = true, std::enable_if_t<enabler&&(DoF>index), int> = 0> \
    __device__ __host__ __forceinline__ ftype& name() { return params[index];}
    DEFINE_ACCESSOR(k1, 0)
    DEFINE_ACCESSOR(k2, 1)
    DEFINE_ACCESSOR(p1, 2)
    DEFINE_ACCESSOR(p2, 3)
    DEFINE_ACCESSOR(k3, 4)
#undef DEFINE_ACCESSOR

    __device__ __host__ __forceinline__
    void update(const kmat<ftype, DoF>& delta){
        for (uint32_t i = 0; i < DoF; i++){
            params[i] += delta[i];
        }
    }
    __device__ __host__ __forceinline__
    ftype squaredNorm() const {
        return kmat<ftype, DoF>{params}.sqrNorm();
    }

    struct ValueJacobian{
        kmat<ftype, 2> value;
        //k1, k2, p1, p2, k3
        kmat<ftype, 2, nbParams> distortJac;
        kmat<ftype, 2, 2> normXYJac;
    };
    __device__ __host__ __forceinline__
    ValueJacobian computeValueJacobian(const kmat<ftype, 2>& normXY) const;

    bool operator==(const DistortParams<ftype, nbParams>& other) const{
        return params == other.params;
    }
};

template <typename ftype, uint32_t nbParams>
__device__ __host__ __forceinline__
typename DistortParams<ftype, nbParams>::ValueJacobian
DistortParams<ftype, nbParams>::computeValueJacobian(const kmat<ftype, 2>& normXY) const
{
    const auto& k1 = this->k1();
    const auto& k2 = this->k2();
    const auto& p1 = this->p1();
    const auto& p2 = this->p2();
    const auto& k3 = this->k3();
    ValueJacobian ret;
    const ftype r2 = normXY.sqrNorm();
    const ftype x0 = normXY[0], x1 = normXY[1];
    const ftype scale = 1 + k1 * r2 + k2 * (r2 * r2) + k3 * (r2 * r2 * r2);
    ret.value = normXY * scale
                + kmat<ftype, 2>({
                                         ftype(2*p1*(x0*x1) + p2*(r2 + 2*sqr(x0))),
                                         ftype(2*p2*(x0*x1) + p1*(r2 + 2*sqr(x1)))
                                 });

    struct{
        kmat<ftype, 2> k1;
        kmat<ftype, 2> k2;
        kmat<ftype, 2> p1;
        kmat<ftype, 2> p2;
        kmat<ftype, 2> k3;
        kmat<ftype, 2, 2> normXY;
    }jacobian;
    jacobian.k1 = normXY * r2;
    jacobian.k2 = normXY * sqr(r2);
    jacobian.p1 = {{2*x0*x1, r2 + 2*sqr(x1)}};
    jacobian.p2 = {{r2 + 2*sqr(x0), 2*x0*x1}};
    jacobian.k3 = normXY * (r2*r2*r2);
    if (nbParams >= 1) {
        ret.distortJac.assignBlock(0,0, jacobian.k1);
    }
    if (nbParams >= 2) {
        ret.distortJac.assignBlock(0,1, jacobian.k2);
    }
    if (nbParams >= 3) {
        ret.distortJac.assignBlock(0,2, jacobian.p1);
    }
    if (nbParams >= 4) {
        ret.distortJac.assignBlock(0,3, jacobian.p2);
    }
    if (nbParams >= 5) {
        ret.distortJac.assignBlock(0,4, jacobian.k3);
    }
    {
        const auto scaleJacR2 = k1 + 2*k2*r2 + 3*k3*(r2*r2);
        const auto j01 = 2*x0*x1*scaleJacR2 + 2*p1*x0 + 2*p2*x1;
        jacobian.normXY = kmat<ftype, 2, 2>{
                                   scale + 2*scaleJacR2*sqr(x0) + 2*p1*x1 + 6*p2*x0, ftype(j01),
                                   ftype(j01), scale + 2*scaleJacR2*sqr(x1) + 6*p1*x1 + 2*p2*x0
                           };
    }
    ret.normXYJac = jacobian.normXY;
    return ret;
}

//using ftype = FPTraitsSSD::lpf; template <uint32_t nbDistortParams>
template <typename ftype, uint32_t nbDistortParams>
__device__ __host__ __forceinline__
typename DistortParams<ftype, nbDistortParams>::ValueJacobian computeDistortValueJacobian(
        const kmat<ftype, 2>& normXY, const DistortParams<ftype, nbDistortParams>& distortParams){
    return distortParams.computeValueJacobian(normXY);
}

//********************Intrinsic Parameters**************************

//using ftype = float;
template <typename ftype_, bool rolling_>
struct IntrinsicsF2C2D5 {
    using ftype = ftype_;
    static constexpr uint32_t nbFParams = 2;
    static constexpr uint32_t nbCParams = 2;
    static constexpr uint32_t nbFCParams = nbFParams + nbCParams;
    static constexpr uint32_t nbDistortParams = 5;
    static constexpr uint32_t DoF = nbFCParams + nbDistortParams;
	static constexpr bool rolling = rolling_;
    ftype f[2];// x, y
    ftype c[2];// x, y
    DistortParams<ftype, nbDistortParams> d;
	std::conditional_t<rolling, ftype, std::monostate> rollingCenter;

    __device__ __host__ __forceinline__
    ftype fx() const {return f[0];}
    __device__ __host__ __forceinline__
    ftype fy() const {return f[1];}
    __device__ __host__ __forceinline__
    ftype cx() const {return c[0];}
    __device__ __host__ __forceinline__
    ftype cy() const {return c[1];}

    __device__ __host__ __forceinline__
    ftype squaredNorm() const {
        return sqr(f[0]) + sqr(f[1]) + sqr(c[0]) + sqr(c[1]) + d.squaredNorm();
    }

    __device__ __host__ __forceinline__
    void update(const kmat<ftype, DoF> &delta){
        for (int i = 0; i < 2; i++){
            f[i] += delta[i];
            c[i] += delta[2 + i];
        }
        d.update(kmat<ftype, nbDistortParams>{delta.data()  + nbFCParams});
    }

    template <typename FType>
    __device__ __host__ __forceinline__
    void update(const kmat<FType, DoF>& delta){
        update(delta.template cast<ftype>());
    }

    struct ValueJacobian{
        kmat<ftype, 2> value;
        struct {
            //fx, fy, cx, cy, k1, k2, p1, p2, k3
            kmat<ftype, 2, DoF> intri;
            kmat<ftype, 2, 3> pt;
        } jacobian;
    };
    __device__ __host__ __forceinline__
    ValueJacobian computeValueDerivative(const kmat<ftype, 3>& pt) const
    {
        const ftype x = pt[0], y = pt[1], z = pt[2];
        const ftype z_inv = fast_rcp(z);
        const kmat<ftype, 2> normXY({x * z_inv, y * z_inv});
        const kmat<ftype, 2, 3> normXYJacXYZ({
                                                     z_inv, 0, -x * sqr(z_inv),
                                                     0, z_inv, -y * sqr(z_inv)
                                             });
        const typename DistortParams<ftype, nbDistortParams>::ValueJacobian distValJac = computeDistortValueJacobian(normXY);
        const ftype fx = this->fx(), fy = this->fy();
        const ftype cx = this->cx(), cy = this->cy();
        ValueJacobian ret;
        ret.value = {distValJac.value[0] * fx + cx, distValJac.value[1] * fy + cy};

        const kmat<ftype, 2, 2> fJac({
                                             distValJac.value[0], 0,
                                             0, distValJac.value[1]
                                     });
        ret.jacobian.intri.assignBlock(0,0, fJac);
        ret.jacobian.intri.assignBlock(0,2, kmat<ftype, 2, 2>::eye());
        ret.jacobian.intri.assignBlock(0u,nbFCParams, distValJac.distortJac.row(0) * fx);
        ret.jacobian.intri.assignBlock(1u,nbFCParams, distValJac.distortJac.row(1) * fy);

//        ret.jacobian.pt = kmat<ftype,2,2>({fx,0,0,fy}) * distValJac.normXYJac * normXYJacXYZ;
        ret.jacobian.pt.assignBlock(0,0, distValJac.normXYJac.row(0) * (fx * z_inv));
        ret.jacobian.pt.assignBlock(1,0, distValJac.normXYJac.row(1) * (fy * z_inv));
        ret.jacobian.pt.assignBlock(0,2, distValJac.normXYJac * kmat<ftype, 2, 1>({
                                                                                          -fx * x * sqr(z_inv),
                                                                                          -fy * y * sqr(z_inv)}));
        return ret;
    }
#ifndef UNIT_TEST
private:
#endif
    __device__ __host__ __forceinline__
    DistortParams<ftype, nbDistortParams> getDistortParams() const { return d; }
    __device__ __host__ __forceinline__
    typename DistortParams<ftype, nbDistortParams>::ValueJacobian
    computeDistortValueJacobian(const kmat<ftype, 2>& normXY) const{
        return ::rba::computeDistortValueJacobian(normXY, getDistortParams());
    }
public:
    constexpr bool operator==(const IntrinsicsF2C2D5& other) const {
        return f[0] == other.f[0] && f[1] == other.f[1]
               && c[0] == other.c[0] && c[1] == other.c[1]
               && d == other.d && rollingCenter == other.rollingCenter;
    }
}; // struct IntrinsicsF2C2D5

// using ftype = float;
// constexpr bool rolling_ = false;
template <typename ftype_, bool rolling_>
struct IntrinsicsF2D5 {
    using ftype = ftype_;
    static constexpr uint32_t nbFParams = 2;
    static constexpr uint32_t nbCParams = 0;
    static constexpr uint32_t nbFCParams = nbFParams + nbCParams;
    static constexpr uint32_t nbDistortParams = 5;
    static constexpr uint32_t DoF = nbFCParams + nbDistortParams;
	static constexpr bool rolling = rolling_;
    ftype f[2];// x, y
    DistortParams<ftype, nbDistortParams> d;
	std::conditional_t<rolling, ftype, std::monostate> rollingCenter;

    __device__ __host__ __forceinline__
    ftype fx() const {return f[0];}
    __device__ __host__ __forceinline__
    ftype fy() const {return nbFParams >= 2 ? f[1] : fx();}
    __device__ __host__ __forceinline__
    ftype cx() const {return 0;}
    __device__ __host__ __forceinline__
    ftype cy() const {return 0;}

    __device__ __host__ __forceinline__
    ftype squaredNorm() const {
        return sqr(f[0]) + sqr(f[1]) + d.squaredNorm();
    }

    __device__ __host__ __forceinline__
    void update(const kmat<ftype, DoF> &delta){
        for (uint32_t i = 0; i < nbFParams; i++){
            f[i] += delta[i];
        }
        d.update(kmat<ftype, nbDistortParams>{delta.data()  + nbFCParams});
    }

    template <typename FType>
    __device__ __host__ __forceinline__
    void update(const kmat<FType, DoF>& delta){
        update(delta.template cast<ftype>());
    }

    struct ValueJacobian{
        kmat<ftype, 2> value;
        struct {
            //fx, fy, cx, cy, k1, k2, p1, p2, k3
            kmat<ftype, 2, DoF> intri;
            kmat<ftype, 2, 3> pt;
        } jacobian;
    };
    __device__ __host__ __forceinline__
    ValueJacobian computeValueDerivative(const kmat<ftype, 3>& pt) const
    {
        const ftype x = pt[0], y = pt[1], z = pt[2];
        const ftype z_inv = fast_rcp(z);
        const kmat<ftype, 2> normXY({x * z_inv, y * z_inv});
        const kmat<ftype, 2, 3> normXYJacXYZ({
                                                     z_inv, 0, -x * sqr(z_inv),
                                                     0, z_inv, -y * sqr(z_inv)
                                             });
        const typename DistortParams<ftype, nbDistortParams>::ValueJacobian distValJac = computeDistortValueJacobian(normXY);
        const ftype fx = this->fx(), fy = this->fy();
        const ftype cx = this->cx(), cy = this->cy();
        ValueJacobian ret;
        ret.value = {distValJac.value[0] * fx + cx, distValJac.value[1] * fy + cy};

        const kmat<ftype, 2, 2> fJac({
                                             distValJac.value[0], 0,
                                             0, distValJac.value[1]
                                     });
        ret.jacobian.intri.assignBlock(0,0, fJac);
        ret.jacobian.intri.assignBlock(0u,nbFCParams, distValJac.distortJac.row(0) * fx);
        ret.jacobian.intri.assignBlock(1u,nbFCParams, distValJac.distortJac.row(1) * fy);

//        ret.jacobian.pt = kmat<ftype,2,2>({fx,0,0,fy}) * distValJac.normXYJac * normXYJacXYZ;
        ret.jacobian.pt.assignBlock(0,0, distValJac.normXYJac.row(0) * (fx * z_inv));
        ret.jacobian.pt.assignBlock(1,0, distValJac.normXYJac.row(1) * (fy * z_inv));
        ret.jacobian.pt.assignBlock(0,2, distValJac.normXYJac * kmat<ftype, 2, 1>({
                                                                                          -fx * x * sqr(z_inv),
                                                                                          -fy * y * sqr(z_inv)}));
        return ret;
    }
#ifndef UNIT_TEST
private:
#endif
    __device__ __host__ __forceinline__
    DistortParams<ftype, nbDistortParams> getDistortParams() const { return d; }
    __device__ __host__ __forceinline__
    typename DistortParams<ftype, nbDistortParams>::ValueJacobian
    computeDistortValueJacobian(const kmat<ftype, 2>& normXY) const{
        return ::rba::computeDistortValueJacobian(normXY, getDistortParams());
    }
public:
    constexpr bool operator==(const IntrinsicsF2D5& other) const {
        return f[0] == other.f[0] && f[1] == other.f[1]
               && d == other.d && rollingCenter == other.rollingCenter;
    }
}; // struct IntrinsicsF2D5

template <typename ftype_, bool rolling_>
struct IntrinsicsF1C2D5 {
    using ftype = ftype_;
    static constexpr uint32_t nbFParams = 1;
    static constexpr uint32_t nbCParams = 2;
    static constexpr uint32_t nbFCParams = nbFParams + nbCParams;
    static constexpr uint32_t nbDistortParams = 5;
    static constexpr uint32_t DoF = nbFCParams + nbDistortParams;
	static constexpr bool rolling = rolling_;
    ftype f;
    ftype c[2];// x, y
    DistortParams<ftype, nbDistortParams> d;
	std::conditional_t<rolling, ftype, std::monostate> rollingCenter;

    __device__ __host__ __forceinline__
    ftype fx() const {return f;}
    __device__ __host__ __forceinline__
    ftype fy() const {return f;}
    __device__ __host__ __forceinline__
    ftype cx() const {return c[0];}
    __device__ __host__ __forceinline__
    ftype cy() const {return c[1];}

    __device__ __host__ __forceinline__
    ftype squaredNorm() const {
        return sqr(f) + sqr(c[0]) + sqr(c[1]) + d.squaredNorm();
    }

    __device__ __host__ __forceinline__
    void update(const kmat<ftype,DoF> &delta){
        f += delta[0];
        for (int i = 0; i < 2; i++){
            c[i] += delta[1 + i];
        }
        d.update(kmat<ftype, nbDistortParams>{delta.data()  + nbFCParams});
    }

    template <typename FType>
    __device__ __host__ __forceinline__
    void update(const kmat<FType, DoF>& delta){
        update(delta.template cast<ftype>());
    }

    struct ValueJacobian{
        kmat<ftype, 2> value;
        struct {
            //f, k1, k2
            kmat<ftype, 2, DoF> intri;
            kmat<ftype, 2, 3> pt;
        } jacobian;
    };
    __device__ __host__ __forceinline__
    ValueJacobian computeValueDerivative(const kmat<ftype, 3>& pt) const
    {
        const ftype x = pt[0], y = pt[1], z = pt[2];
        const ftype z_inv = fast_rcp(z);
        const kmat<ftype, 2> normXY({x * z_inv, y * z_inv});
        const kmat<ftype, 2, 3> normXYJacXYZ({
                                                     z_inv, 0, -x * sqr(z_inv),
                                                     0, z_inv, -y * sqr(z_inv)
                                             });
        const typename DistortParams<ftype, nbDistortParams>::ValueJacobian distValJac = computeDistortValueJacobian(normXY);
        const ftype fx = this->fx(), fy = this->fy();
        const ftype cx = this->cx(), cy = this->cy();
        ValueJacobian ret;
        ret.value = {distValJac.value[0] * fx + cx, distValJac.value[1] * fy + cy};

        const kmat<ftype, 2> fJac({distValJac.value[0], distValJac.value[1]});
        ret.jacobian.intri.assignBlock(0,0, fJac);
        ret.jacobian.intri.assignBlock(0,1, kmat<ftype, 2, 2>::eye());
        ret.jacobian.intri.assignBlock(0u,nbFCParams, distValJac.distortJac.row(0) * fx);
        ret.jacobian.intri.assignBlock(1u,nbFCParams, distValJac.distortJac.row(1) * fy);

//        ret.jacobian.pt = kmat<ftype,2,2>({fx,0,0,fy}) * distValJac.normXYJac * normXYJacXYZ;
        ret.jacobian.pt.assignBlock(0,0, distValJac.normXYJac.row(0) * (fx * z_inv));
        ret.jacobian.pt.assignBlock(1,0, distValJac.normXYJac.row(1) * (fy * z_inv));
        ret.jacobian.pt.assignBlock(0,2, distValJac.normXYJac * kmat<ftype, 2, 1>({
                                                                                          -fx * x * sqr(z_inv),
                                                                                          -fy * y * sqr(z_inv)}));
        return ret;
    }
#ifndef UNIT_TEST
private:
#endif
    __device__ __host__ __forceinline__
    DistortParams<ftype, nbDistortParams> getDistortParams() const { return d; }
    __device__ __host__ __forceinline__
    typename DistortParams<ftype, nbDistortParams>::ValueJacobian
    computeDistortValueJacobian(const kmat<ftype, 2>& normXY) const{
        return ::rba::computeDistortValueJacobian(normXY, getDistortParams());
    }
public:
    constexpr bool operator==(const IntrinsicsF1C2D5& other) const {
        return f == other.f
               && c[0] == other.c[0] && c[1] == other.c[1]
               && d == other.d && rollingCenter == other.rollingCenter;
    }
}; // struct IntrinsicsF1C2D5


// using ftype_ = float;
// constexpr bool rolling_ = false;
template <typename ftype_, bool rolling_>
struct IntrinsicsF1D5 {
    using ftype = ftype_;
    static constexpr uint32_t nbFParams = 1;
    static constexpr uint32_t nbCParams = 0;
    static constexpr uint32_t nbFCParams = nbFParams + nbCParams;
    static constexpr uint32_t nbDistortParams = 5;
    static constexpr uint32_t DoF = nbFCParams + nbDistortParams;
	static constexpr bool rolling = rolling_;
    ftype f;
    DistortParams<ftype, nbDistortParams> d;
	std::conditional_t<rolling, ftype, std::monostate> rollingCenter;

    __device__ __host__ __forceinline__
    ftype fx() const {return f;}
    __device__ __host__ __forceinline__
    ftype fy() const {return f;}
    __device__ __host__ __forceinline__
    ftype cx() const {return 0;}
    __device__ __host__ __forceinline__
    ftype cy() const {return 0;}

    __device__ __host__ __forceinline__
    ftype squaredNorm() const {
        return sqr(f) + d.squaredNorm();
    }

    __device__ __host__ __forceinline__
    void update(const kmat<ftype,DoF> &delta){
        f += delta[0];
        d.update(kmat<ftype, nbDistortParams>{delta.data()  + nbFCParams});
    }

    template <typename FType>
    __device__ __host__ __forceinline__
    void update(const kmat<FType, DoF>& delta){
        update(delta.template cast<ftype>());
    }

    struct ValueJacobian{
        kmat<ftype, 2> value;
        struct {
            //f, k1, k2
            kmat<ftype, 2, DoF> intri;
            kmat<ftype, 2, 3> pt;
        } jacobian;
    };
    __device__ __host__ __forceinline__
    ValueJacobian computeValueDerivative(const kmat<ftype, 3>& pt) const
    {
        const ftype x = pt[0], y = pt[1], z = pt[2];
        const ftype z_inv = fast_rcp(z);
        const kmat<ftype, 2> normXY({x * z_inv, y * z_inv});
        const kmat<ftype, 2, 3> normXYJacXYZ({
                                                     z_inv, 0, -x * sqr(z_inv),
                                                     0, z_inv, -y * sqr(z_inv)
                                             });
        const typename DistortParams<ftype, nbDistortParams>::ValueJacobian distValJac = computeDistortValueJacobian(normXY);
        const ftype fx = this->fx(), fy = this->fy();
        const ftype cx = this->cx(), cy = this->cy();
        ValueJacobian ret;
        ret.value = {distValJac.value[0] * fx + cx, distValJac.value[1] * fy + cy};

        const kmat<ftype, 2> fJac({distValJac.value[0], distValJac.value[1]});
        ret.jacobian.intri.assignBlock(0,0, fJac);
        ret.jacobian.intri.assignBlock(0u,nbFCParams, distValJac.distortJac.row(0) * fx);
        ret.jacobian.intri.assignBlock(1u,nbFCParams, distValJac.distortJac.row(1) * fy);

//        ret.jacobian.pt = kmat<ftype,2,2>({fx,0,0,fy}) * distValJac.normXYJac * normXYJacXYZ;
        ret.jacobian.pt.assignBlock(0,0, distValJac.normXYJac.row(0) * (fx * z_inv));
        ret.jacobian.pt.assignBlock(1,0, distValJac.normXYJac.row(1) * (fy * z_inv));
        ret.jacobian.pt.assignBlock(0,2, distValJac.normXYJac * kmat<ftype, 2, 1>({
                                                                                          -fx * x * sqr(z_inv),
                                                                                          -fy * y * sqr(z_inv)}));
        return ret;
    }
#ifndef UNIT_TEST
private:
#endif
    __device__ __host__ __forceinline__
    DistortParams<ftype, nbDistortParams> getDistortParams() const { return d; }
    __device__ __host__ __forceinline__
    typename DistortParams<ftype, nbDistortParams>::ValueJacobian
    computeDistortValueJacobian(const kmat<ftype, 2>& normXY) const{
        return ::rba::computeDistortValueJacobian(normXY, getDistortParams());
    }
public:
    constexpr bool operator==(const IntrinsicsF1D5& other) const {
        return f == other.f
               && d == other.d && rollingCenter == other.rollingCenter;
    }
}; // struct IntrinsicsF1D5

// If the fixed cx/cy/p1/p2/p3 parameters are non-zero, they should be applied to observation to make these parameters zero.
//using ftype = float;
template <typename ftype_, bool rolling_>
struct IntrinsicsF1D2 {
    using ftype = ftype_;
    static constexpr uint32_t nbFParams = 1;
    static constexpr uint32_t nbCParams = 0;
    static constexpr uint32_t nbFCParams = nbFParams + nbCParams;
    static constexpr uint32_t nbDistortParams = 2;
    static constexpr uint32_t DoF = nbFCParams + nbDistortParams;
	static constexpr bool rolling = rolling_;
    ftype f;
    DistortParams<ftype, nbDistortParams> d;
	std::conditional_t<rolling, ftype, std::monostate> rollingCenter;

    __device__ __host__ __forceinline__
    ftype fx() const {return f;}
    __device__ __host__ __forceinline__
    ftype fy() const {return f;}
    __device__ __host__ __forceinline__
    ftype cx() const {return 0;}
    __device__ __host__ __forceinline__
    ftype cy() const {return 0;}

    __device__ __host__ __forceinline__
    ftype squaredNorm() const {
        return sqr(f) + d.squaredNorm();
    }

    template <typename T>
    __device__ __host__ __forceinline__
    void update(const kmat<T, DoF>& delta){
        f += static_cast<ftype>(delta[0]);
        d.update(kmat<T, nbDistortParams>{delta.data()  + nbFCParams}.template cast<ftype>());
    }

    struct ValueJacobian{
        kmat<ftype, 2> value;
        struct {
            //f, k1, k2
            kmat<ftype, 2, DoF> intri;
            kmat<ftype, 2, 3> pt;
        } jacobian;
    };
    __device__ __host__ __forceinline__
    ValueJacobian computeValueDerivative(const kmat<ftype, 3>& pt) const
    {
        const ftype x0 = pt[0], x1 = pt[1], x2 = pt[2];
        const ftype x2inv = fast_rcp(x2);
        const kmat<ftype, 2> normXY({x0*x2inv, x1*x2inv});
        const kmat<ftype, 2, 3> normXYJacXYZ({
                                                     x2inv, 0, -x0*sqr(x2inv),
                                                     0, x2inv, -x1*sqr(x2inv)
                                             });
        const typename DistortParams<ftype, nbDistortParams>::ValueJacobian distValJac = computeDistortValueJacobian(normXY);
        const ftype fx = f, fy = f;
        ValueJacobian ret{};
        ret.value = {distValJac.value[0] * fx, distValJac.value[1] * fy};

        const kmat<ftype, 2, 1> fJac({
                                             distValJac.value[0],
                                             distValJac.value[1]});
        ret.jacobian.intri.assignBlock(0,0, fJac);
        ret.jacobian.intri.assignBlock(0u,nbFCParams, distValJac.distortJac.row(0) * fx);
        ret.jacobian.intri.assignBlock(1u,nbFCParams, distValJac.distortJac.row(1) * fy);

//        ret.jacobian.pt = kmat<ftype,2,2>({fx,0,0,fy}) * distValJac.normXYJac * normXYJacXYZ;
        ret.jacobian.pt.assignBlock(0,0, distValJac.normXYJac.row(0) * (fx*x2inv));
        ret.jacobian.pt.assignBlock(1,0, distValJac.normXYJac.row(1) * (fy*x2inv));
        ret.jacobian.pt.assignBlock(0,2, distValJac.normXYJac * kmat<ftype, 2, 1>({
                                                                                          -fx*x0*sqr(x2inv),
                                                                                          -fy*x1*sqr(x2inv)}));
        return ret;
    }
#ifndef UNIT_TEST
    private:
#endif
    __device__ __host__ __forceinline__
    DistortParams<ftype, nbDistortParams> getDistortParams() const { return d; }
    __device__ __host__ __forceinline__
    typename DistortParams<ftype, nbDistortParams>::ValueJacobian
    computeDistortValueJacobian(const kmat<ftype, 2>& normXY) const{
        return ::rba::computeDistortValueJacobian(normXY, getDistortParams());
    }
public:
    constexpr bool operator==(const IntrinsicsF1D2& other) const {
        return f == other.f && d == other.d && rollingCenter == other.rollingCenter;
    }
}; // struct IntrinsicsF1D2

// If the fixed cx/cy/p1/p2/p3 parameters are non-zero, they should be applied to observation to make these parameters zero.
//using ftype = float;
template <typename ftype_, bool rolling_>
struct IntrinsicsF1 {
    using ftype = ftype_;
    static constexpr uint32_t nbFParams = 1;
    static constexpr uint32_t nbCParams = 0;
    static constexpr uint32_t nbFCParams = nbFParams + nbCParams;
    static constexpr uint32_t nbDistortParams = 0;
    static constexpr uint32_t DoF = nbFCParams + nbDistortParams;
	static constexpr bool rolling = rolling_;
    ftype f;
    DistortParams<ftype, nbDistortParams> d;
	std::conditional_t<rolling, ftype, std::monostate> rollingCenter;

    __device__ __host__ __forceinline__
    ftype fx() const {return f;}
    __device__ __host__ __forceinline__
    ftype fy() const {return f;}
    __device__ __host__ __forceinline__
    ftype cx() const {return 0;}
    __device__ __host__ __forceinline__
    ftype cy() const {return 0;}

    __device__ __host__ __forceinline__
    ftype squaredNorm() const {
        return sqr(f) + d.squaredNorm();
    }

    template <typename T>
    __device__ __host__ __forceinline__
    void update(const kmat<T, DoF>& delta){
        f += static_cast<ftype>(delta[0]);
        d.update(kmat<T, nbDistortParams>{delta.data() + nbFCParams}.template cast<ftype>());
    }

    struct ValueJacobian{
        kmat<ftype, 2> value;
        struct {
            //f, k1, k2
            kmat<ftype, 2, DoF> intri;
            kmat<ftype, 2, 3> pt;
        } jacobian;
    };
    __device__ __host__ __forceinline__
    ValueJacobian computeValueDerivative(const kmat<ftype, 3>& pt) const
    {
        const ftype x0 = pt[0], x1 = pt[1], x2 = pt[2];
        const ftype x2inv = fast_rcp(x2);
        const kmat<ftype, 2> normXY({x0*x2inv, x1*x2inv});
        const kmat<ftype, 2, 3> normXYJacXYZ({
                                                     x2inv, 0, -x0*sqr(x2inv),
                                                     0, x2inv, -x1*sqr(x2inv)
                                             });
        const typename DistortParams<ftype, nbDistortParams>::ValueJacobian distValJac = computeDistortValueJacobian(normXY);
        const ftype fx = f, fy = f;
        ValueJacobian ret{};
        ret.value = {distValJac.value[0] * fx, distValJac.value[1] * fy};

        const kmat<ftype, 2, 1> fJac({
                                             distValJac.value[0],
                                             distValJac.value[1]});
        ret.jacobian.intri.assignBlock(0,0, fJac);
        ret.jacobian.intri.assignBlock(0u,nbFCParams, distValJac.distortJac.row(0) * fx);
        ret.jacobian.intri.assignBlock(1u,nbFCParams, distValJac.distortJac.row(1) * fy);

//        ret.jacobian.pt = kmat<ftype,2,2>({fx,0,0,fy}) * distValJac.normXYJac * normXYJacXYZ;
        ret.jacobian.pt.assignBlock(0,0, distValJac.normXYJac.row(0) * (fx*x2inv));
        ret.jacobian.pt.assignBlock(1,0, distValJac.normXYJac.row(1) * (fy*x2inv));
        ret.jacobian.pt.assignBlock(0,2, distValJac.normXYJac * kmat<ftype, 2, 1>({
                                                                                          -fx*x0*sqr(x2inv),
                                                                                          -fy*x1*sqr(x2inv)}));
        return ret;
    }
#ifndef UNIT_TEST
    private:
#endif
    __device__ __host__ __forceinline__
    DistortParams<ftype, nbDistortParams> getDistortParams() const { return d; }
    __device__ __host__ __forceinline__
    typename DistortParams<ftype, nbDistortParams>::ValueJacobian
    computeDistortValueJacobian(const kmat<ftype, 2>& normXY) const{
        return ::rba::computeDistortValueJacobian(normXY, getDistortParams());
    }
public:
    constexpr bool operator==(const IntrinsicsF1<ftype, rolling>& other) const {
        return f == other.f && d == other.d && rollingCenter == other.rollingCenter;
    }
}; // struct IntrinsicsF1

//using ftype = float;
template <typename ftype_, bool rolling_>
struct IntrinsicsF2C2 {
    using ftype = ftype_;
    static constexpr uint32_t nbFParams = 2;
    static constexpr uint32_t nbCParams = 2;
    static constexpr uint32_t nbFCParams = nbFParams + nbCParams;
    static constexpr uint32_t nbDistortParams = 0;
    static constexpr uint32_t DoF = nbFCParams + nbDistortParams;
	static constexpr bool rolling = rolling_;
    ftype f[2];// x, y
    ftype c[2];// x, y
    DistortParams<ftype, nbDistortParams> d;
	std::conditional_t<rolling, ftype, std::monostate> rollingCenter;

    __device__ __host__ __forceinline__
    ftype fx() const {return f[0];}
    __device__ __host__ __forceinline__
    ftype fy() const {return f[1];}
    __device__ __host__ __forceinline__
    ftype cx() const {return c[0];}
    __device__ __host__ __forceinline__
    ftype cy() const {return c[1];}

    __device__ __host__ __forceinline__
    ftype squaredNorm() const {
        return sqr(f[0]) + sqr(f[1]) + sqr(c[0]) + sqr(c[1]) + d.squaredNorm();
    }

    __device__ __host__ __forceinline__
    void update(const kmat<ftype, DoF> &delta){
        for (int i = 0; i < 2; i++){
            f[i] += delta[i];
            c[i] += delta[2 + i];
        }
        d.update(kmat<ftype, nbDistortParams>{delta.data() + nbFCParams});
    }

    template <typename FType>
    __device__ __host__ __forceinline__
    void update(const kmat<FType, DoF>& delta){
        update(delta.template cast<ftype>());
    }

    struct ValueJacobian{
        kmat<ftype, 2> value;
        struct {
            //fx, fy, cx, cy
            kmat<ftype, 2, DoF> intri;
            kmat<ftype, 2, 3> pt;
        } jacobian;
    };
    __device__ __host__ __forceinline__
    ValueJacobian computeValueDerivative(const kmat<ftype, 3>& pt) const
    {
        const ftype x = pt[0], y = pt[1], z = pt[2];
        const ftype z_inv = fast_rcp(z);
        const kmat<ftype, 2> normXY({x * z_inv, y * z_inv});
        const typename DistortParams<ftype, nbDistortParams>::ValueJacobian distValJac = computeDistortValueJacobian(normXY);
        const ftype fx = this->fx(), fy = this->fy();
        const ftype cx = this->cx(), cy = this->cy();
        ValueJacobian ret;
        ret.value = {distValJac.value[0] * fx + cx, distValJac.value[1] * fy + cy};

        const kmat<ftype, 2, 2> fJac({
                                             distValJac.value[0], 0,
                                             0, distValJac.value[1]
                                     });
        ret.jacobian.intri.assignBlock(0,0, fJac);
        ret.jacobian.intri.assignBlock(0,2, kmat<ftype, 2, 2>::eye());
        ret.jacobian.intri.assignBlock(0u,nbFCParams, distValJac.distortJac.row(0) * fx);
        ret.jacobian.intri.assignBlock(1u,nbFCParams, distValJac.distortJac.row(1) * fy);
#if 0
        const kmat<ftype, 2, 3> normXYJacXYZ({
            z_inv, 0, -x * sqr(z_inv),
            0, z_inv, -y * sqr(z_inv)
        });
        ret.jacobian.pt = kmat<ftype,2,2>({fx,0,0,fy}) * distValJac.normXYJac * normXYJacXYZ;
#else
        ret.jacobian.pt.assignBlock(0,0, distValJac.normXYJac.row(0) * (fx * z_inv));
        ret.jacobian.pt.assignBlock(1,0, distValJac.normXYJac.row(1) * (fy * z_inv));
        ret.jacobian.pt.assignBlock(0,2, distValJac.normXYJac * kmat<ftype, 2, 1>({
                                                                                          -fx * x * sqr(z_inv),
                                                                                          -fy * y * sqr(z_inv)}));
#endif
        return ret;
    }
#ifndef UNIT_TEST
private:
#endif
    __device__ __host__ __forceinline__
    DistortParams<ftype, nbDistortParams> getDistortParams() const { return d; }
    __device__ __host__ __forceinline__
    typename DistortParams<ftype, nbDistortParams>::ValueJacobian
    computeDistortValueJacobian(const kmat<ftype, 2>& normXY) const{
        return ::rba::computeDistortValueJacobian(normXY, getDistortParams());
    }
public:
    constexpr bool operator==(const IntrinsicsF2C2& other) const {
        return f[0] == other.f[0] && f[1] == other.f[1]
               && c[0] == other.c[0] && c[1] == other.c[1]
               && d == other.d && rollingCenter == other.rollingCenter;
    }
}; // struct IntrinsicsF2C2

//using ftype = float;
template <typename ftype_, bool rolling_>
struct IntrinsicsF2 {
    using ftype = ftype_;
    static constexpr uint32_t nbFParams = 2;
    static constexpr uint32_t nbCParams = 0;
    static constexpr uint32_t nbFCParams = nbFParams + nbCParams;
    static constexpr uint32_t nbDistortParams = 0;
    static constexpr uint32_t DoF = nbFCParams + nbDistortParams;
	static constexpr bool rolling = rolling_;
    ftype f[2];// x, y
    DistortParams<ftype, nbDistortParams> d;
	std::conditional_t<rolling, ftype, std::monostate> rollingCenter;

    __device__ __host__ __forceinline__
    ftype fx() const {return f[0];}
    __device__ __host__ __forceinline__
    ftype fy() const {return f[1];}
    __device__ __host__ __forceinline__
    ftype cx() const {return 0;}
    __device__ __host__ __forceinline__
    ftype cy() const {return 0;}

    __device__ __host__ __forceinline__
    ftype squaredNorm() const {
        return sqr(f[0]) + sqr(f[1]) + d.squaredNorm();
    }

    __device__ __host__ __forceinline__
    void update(const kmat<ftype, DoF> &delta){
        for (int i = 0; i < 2; i++){
            f[i] += delta[i];
        }
        d.update(kmat<ftype, nbDistortParams>{delta.data() + nbFCParams});
    }

    template <typename FType>
    __device__ __host__ __forceinline__
    void update(const kmat<FType, DoF>& delta){
        update(delta.template cast<ftype>());
    }

    struct ValueJacobian{
        kmat<ftype, 2> value;
        struct {
            //fx, fy
            kmat<ftype, 2, DoF> intri;
            kmat<ftype, 2, 3> pt;
        } jacobian;
    };
    __device__ __host__ __forceinline__
    ValueJacobian computeValueDerivative(const kmat<ftype, 3>& pt) const
    {
        const ftype x = pt[0], y = pt[1], z = pt[2];
        const ftype z_inv = fast_rcp(z);
        const kmat<ftype, 2> normXY({x * z_inv, y * z_inv});
        const kmat<ftype, 2, 3> normXYJacXYZ({
                                                     z_inv, 0, -x * sqr(z_inv),
                                                     0, z_inv, -y * sqr(z_inv)
                                             });
        const typename DistortParams<ftype, nbDistortParams>::ValueJacobian distValJac = computeDistortValueJacobian(normXY);
        const ftype fx = this->fx(), fy = this->fy();
        const ftype cx = this->cx(), cy = this->cy();
        ValueJacobian ret;
        ret.value = {distValJac.value[0] * fx + cx, distValJac.value[1] * fy + cy};

        const kmat<ftype, 2, 2> fJac({
                                             distValJac.value[0], 0,
                                             0, distValJac.value[1]
                                     });
        ret.jacobian.intri.assignBlock(0,0, fJac);
        static_assert(nbCParams == 0 || nbCParams == 2, "fatal error");
        if (nbCParams == 2) {
            ret.jacobian.intri.assignBlock(0,2, kmat<ftype, 2, 2>::eye());
        }
        ret.jacobian.intri.assignBlock(0u,nbFCParams, distValJac.distortJac.row(0) * fx);
        ret.jacobian.intri.assignBlock(1u,nbFCParams, distValJac.distortJac.row(1) * fy);

//        ret.jacobian.pt = kmat<ftype,2,2>({fx,0,0,fy}) * distValJac.normXYJac * normXYJacXYZ;
        ret.jacobian.pt.assignBlock(0,0, distValJac.normXYJac.row(0) * (fx * z_inv));
        ret.jacobian.pt.assignBlock(1,0, distValJac.normXYJac.row(1) * (fy * z_inv));
        ret.jacobian.pt.assignBlock(0,2, distValJac.normXYJac * kmat<ftype, 2, 1>({
                                                                                          -fx * x * sqr(z_inv),
                                                                                          -fy * y * sqr(z_inv)}));
        return ret;
    }
#ifndef UNIT_TEST
private:
#endif
    __device__ __host__ __forceinline__
    DistortParams<ftype, nbDistortParams> getDistortParams() const { return d; }
    __device__ __host__ __forceinline__
    typename DistortParams<ftype, nbDistortParams>::ValueJacobian
    computeDistortValueJacobian(const kmat<ftype, 2>& normXY) const{
        return ::rba::computeDistortValueJacobian(normXY, getDistortParams());
    }
public:
    constexpr bool operator==(const IntrinsicsF2& other) const {
        return f[0] == other.f[0] && f[1] == other.f[1]
               && d == other.d && rollingCenter == other.rollingCenter;
    }
}; // struct IntrinsicsF2

// Used when intrinsics and extrinsics are 1:1 coupled and merged.
//using ftype = float;
template <typename ftype_>
struct IntrinsicsNull {
    using ftype = ftype_;
    static constexpr uint32_t nbFParams = 0;
    static constexpr uint32_t nbCParams = 0;
    static constexpr uint32_t nbFCParams = nbFParams + nbCParams;
    static constexpr uint32_t nbDistortParams = 0;
    static constexpr uint32_t DoF = nbFCParams + nbDistortParams;

    __device__ __host__ __forceinline__
    void update(const kmat<ftype, DoF> &delta){fail();}
    constexpr bool operator==(const IntrinsicsNull<ftype>& other) const {
        return true;
    }
    static constexpr ftype squaredNorm() {return 0.f;}
}; // struct IntrinsicsNull

}