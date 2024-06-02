#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Experimental code for liquid-liquid phase equilibria.

# This code is copyright (c) 2024 Patrick B Warren (STFC).
# Email: patrick.warren{at}stfc.ac.uk.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

# Solve for liquid-liquid coexistence in a DPD binary mixture
# described by mean-field free energy f = fid + fex where
# fid = ρ1 (ln ρ1 - 1) + ρ2 (ln ρ2 - 1),
# fex = π/30 (A11 ρ1² + 2 A12 ρ1 ρ2 + A22 ρ2²).
# The corresponding chemical potentials are:
# μ1 = ln(ρ1) + π/15*(A11*ρ1 + A12*ρ2),
# μ2 = ln(ρ2) + π/15*(A12*ρ1 + A22*ρ2),
# and pressure p = ρ + π/30*(A11*ρ1**2 + 2*A12*ρ1*ρ2 + A22*ρ2**2),
# where ρ = ρ1 + ρ2.  Some tweaking of the initial guess may be needed
# to encourage the solver to find distinct coexisting state points.

# To impose NPT we solve a quadratic equation for ρ which is
# π/30 [A11 x² + 2 A12 x(1-x) + A22 (1-x)²] ρ² + ρ - p0 = 0,
# on writing the pressure in terms of ρ1, ρ2 = xρ, (1-x)ρ.

# Coexisting state initial guesses:
# -a 25,35,25 -g 0.0001,0.99
# -a 25,35,20 -g 0.0001,0.99

import argparse
import numpy as np
from numpy import pi as π
from numpy import sqrt, exp
from numpy import log as ln
from scipy.optimize import root

parser = argparse.ArgumentParser(description='experimental DPD LLE profiles')
parser.add_argument('-r', '--rho', default=3.0, type=float, help='baseline density, default 3.0')
parser.add_argument('-a', '--allA', default='25,30,20', help='A11, A12, A22, default 25, 30, 20')
parser.add_argument('-g', '--guess', default='0.0001,0.99', help='initial guess x, default 0.01, 0.9')
parser.add_argument('--eps', default=1e-6, type=float, help='tolerance to declare coexistence')
parser.add_argument('-v', '--verbose', action='count', default=0, help='increasing verbosity')
args = parser.parse_args()

# define a kernel K(z) and calculate the approximate value of pi/15

dz = 1e-3

z = np.linspace(-1.0, 1.0, round(2.0/dz)+1, dtype=float)

kernel = π/12.0*(1-z)**3*(1+3*z)
kernel[z<0] = np.flip(kernel[z>0])
πby15 = np.trapz(kernel, dx=dz)

if args.verbose:
    print(f'integrated kernel =\t{πby15}')
    print(f'exact result π/15 =\t{π/15}')

ρ0 = args.rho
A11, A12, A22 = eval(args.allA) # returns a tuple
p0 = ρ0 + 1/2*πby15*A11*ρ0**2

if args.verbose:
    print(f'ρ0, A11, A12, A22 = {ρ0} {A11} {A12} {A22}')
    print(f'pressure fixed at p0 = {p0}')

# Use y = ln(x/(1-x)) which inverts to x = 1/(1+exp(-y)).

def fun(y, p0, A11, A12, A22):
    x = 1 / (1 + exp(-y))
    a = 1/2*πby15*(A11*x**2 + 2*A12*x*(1-x) + A22*(1-x)**2)
    ρ = (sqrt(1 + 4*a*p0) - 1) / (2*a)
    ρ1, ρ2 = x*ρ, (1-x)*ρ
    μ1 = ln(ρ1) + πby15*(A11*ρ1 + A12*ρ2)
    μ2 = ln(ρ2) + πby15*(A12*ρ1 + A22*ρ2)
    return [μ1[1] - μ1[0], μ2[1] - μ2[0]]

x0 = np.array(eval(f'[{args.guess}]')) # mole fractions in the two phases (initial guess)
y0 = ln(x0 / (1-x0))
soln = root(fun, y0, args=(p0, A11, A12, A22))
yb = soln.x
xb = 1 / (1 + exp(-yb)) # final mole fractions in the two phases (a 2-vector)

if args.verbose > 1:
    print('Initial guess =', x0)
    print(soln)

if abs(xb[1]-xb[0]) < args.eps:
    raise ValueError(f'State points likely coalesced, x = {xb[0]}, {xb[1]}')

a = 1/2*πby15*(A11*xb**2 + 2*A12*xb*(1-xb) + A22*(1-xb)**2)
ρb = (sqrt(1 + 4*a*p0) - 1) / (2*a) # a 2-vector containing the total densities
ρ1b, ρ2b = xb*ρb, (1-xb)*ρb # these are 2-vectors containing the coexisting densities
μ1 = ln(ρ1b) + πby15*(A11*ρ1b + A12*ρ2b) # 2-vector, should be equal in coexistence
μ2 = ln(ρ2b) + πby15*(A12*ρ1b + A22*ρ2b) # -- ditto --
p = ρb + 1/2*πby15*(A11*ρ1b**2 + 2*A12*ρ1b*ρ2b + A22*ρ2b**2) # -- ditto --

if args.verbose:
    for v in ['ρb', 'x', 'ρ1b', 'ρ2b', 'μ1', 'μ2', 'p']:
        print(f'{v:>3} =', eval(v))

μ1b, μ2b = [np.mean(μ) for μ in [μ1, μ2]] # consensus 'bulk' values

# create an array z in [-zmax, zmax] and a computational domain
# [-zmax+1, zmax-1] within which the convolution operation is valid.

zmax = 7
Lz = zmax - 1
z = np.linspace(-zmax, zmax, round(2*zmax/dz)+1, dtype=float)
lh_bulk = (z < -Lz)
rh_bulk = (z > Lz)
domain = ~lh_bulk & ~rh_bulk

def initial_density_profile(ρb):
    ρ = np.zeros_like(z)
    ρ[z<0] = ρb[0]
    ρ[z>0] = ρb[1]
    ρ[z==0] = 0.5*(ρb[0] + ρb[1])
    return ρ

def clamp_density_profile(ρ, ρb):
    ρ[lh_bulk] = ρb[0]
    ρ[rh_bulk] = ρb[1]
    return ρ

α = 0.1
tol = 1e-10
max_iters = 10**3

ρ1 = initial_density_profile(ρ1b)
ρ2 = initial_density_profile(ρ2b)

# Solve the following by Picard iteration :
# ρ_i(z) = exp[ μ_i - ∑_j ∫ dz' ρ_j(z') U_ij(z-z') ]

for i in range(max_iters):
    ρ1_kern = dz * np.convolve(ρ1, kernel, mode='same')
    ρ2_kern = dz * np.convolve(ρ2, kernel, mode='same')
    ρ1_new = clamp_density_profile(exp(μ1b - A11*ρ1_kern - A12*ρ2_kern), ρ1b)
    ρ2_new = clamp_density_profile(exp(μ2b - A12*ρ1_kern - A22*ρ2_kern), ρ2b)
    ρ1 = (1-α)*ρ1 + α*ρ1_new
    ρ2 = (1-α)*ρ2 + α*ρ2_new
    int_abs_Δρ1 = np.trapz(np.abs(ρ1_new-ρ1), dx=dz)
    int_abs_Δρ2 = np.trapz(np.abs(ρ2_new-ρ2), dx=dz)
    if int_abs_Δρ1 + int_abs_Δρ2 < tol: # early escape if converged
        break

import matplotlib.pyplot as plt

region = ~(z<-3) & ~(z>3)

plt.plot(z[region], ρ1[region])
plt.plot(z[region], ρ2[region])

plt.show()
