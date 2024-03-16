#/usr/bin/env python3
from sympy import *

g0, g1, g2, t0, t1, t2 = symbols('g0, g1, g2, t0, t1, t2', real=True, commutative=True)
g = Matrix([g0, g1, g2])
t = Matrix([t0, t1, t2])

def gvec2mat(g0, g1, g2):
    r = Matrix.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            #@fixme: why [j,i] here? we've changed in C++ code.
            r[j,i] = ((1 - g.dot(g)) * KroneckerDelta(i, j) + 2 * g[i] * g[j] + 2 * sum([LeviCivita(i, j, k) * g[k] for k in range(3)])) / (1 + g.dot(g))
    return r

def quat2mat(w, x, y, z):
    return Matrix([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])


r = gvec2mat(g0, g1, g2)

# R00, R01, R02, R10, R11, R12, R20, R21, R22 = symbols('R00, R01, R02, R10, R11, R12, R20, R21, R22', real=True, commutative=True)
# R = Matrix(3,3, [R00, R01, R02, R10, R11, R12, R20, R21, R22])



##############
x0, x1, x2, fx, fy, cx, cy, k1, k2, p1, p2, k3 = symbols('x0, x1, x2, fx, fy, cx, cy, k1, k2, p1, p2, k3', real=True, commutative=True)
x = Matrix((x0,x1,x2))
(r*x).jacobian([g0,g1,g2]).subs({g0:0,g1:0,g2:0})

q0, q1, q2, q3 = symbols('q0, q1, q2, q3', real=True, commutative=True)
R = quat2mat(q0, q1, q2, q3)

K = Matrix([[fx, 0, cx], [0, fy, cy], [0,0,1]])

transXYZ = R*r*x+t
normalizedXY = Matrix(transXYZ[0:2])/transXYZ[2]
radius=normalizedXY.norm()
distorted = Matrix([normalizedXY[0]*(1+k1*radius**2+k2*radius**4+k3*radius**6)+2*p1*normalizedXY[0]*normalizedXY[1]+p2*(radius**2+2*normalizedXY[0]**2), normalizedXY[1]*(1+k1*radius**2+k2*radius**4) + p1*(radius**2+2*normalizedXY[1]**2) + 2*p2*normalizedXY[0]*normalizedXY[1]])
proj = Matrix([fx*distorted[0]+cx, fy*distorted[1]+cy])

def evalExpr(expr):
    return expr.subs({
        q0:-0.211638, q1:-0.436087, q2:0.732061, q3:0.478670,
        t0:0.297728, t1:-0.079217, t2:-0.408528,
        fx:2473.053955, fy:1615.958984,
        cx:2421.596436, cy:1771.178955,
        k1:0.020521, k2:-0.001724, p1:0.029045, p2:-0.044401, k3:0.000000,
        x0:-1.083088, x1:5.213296, x2:-1.393763,
        g0:0,g1:0,g2:0
    })