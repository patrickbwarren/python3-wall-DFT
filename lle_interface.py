#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Experimental code for density profiles in liquid-liquid phase
# equilibria using a DPD mean-field DFT.

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

# Benchmarks: -A 25,30,20 (default) ; -A 25,50,25 ; -A 25,50,20 --tol=1e-9
# Γ1, Γ2, γ = -0.1060648627316102   -0.1111889714951698   1.3385683713932508
# Γ1, Γ2, γ = -0.20759700220097246  -0.20759700212922425  3.676873998304302
# Γ1, Γ2, γ = -0.2326749744700184   -0.2600528420615002   4.183680028074775

import argparse
import numpy as np
from numpy import pi as π
from numpy import sqrt, exp
from numpy import log as ln
from scipy.optimize import root
from wallDFT import df_header, df_to_agr

parser = argparse.ArgumentParser(description='experimental DPD LLE profiles')
parser.add_argument('--max-iters', default='10^3', help='max number of iterations, default 10^3')
parser.add_argument('--tolerance', default='1e-10', type=float, help='convergence tol, default 1e-10')
parser.add_argument('--alpha', default=0.1, type=float, help='mixing fraction, default 0.1')
parser.add_argument('--zmax', default=6.0, type=float, help='maximum distance in z, default 6.0')
parser.add_argument('--dz', default=1e-3, type=float, help='spacing in z, default 1e-3')
parser.add_argument('--zcut', default=3.0, type=float, help='cut-off in z, default 3.0')
parser.add_argument('--gridz', default=0.02, type=float, help='filter spacing in z, default 0.02')
parser.add_argument('--Abase', default=25.0, type=float, help='baseline A, default 25.0')
parser.add_argument('--ktbyrc2', default=12.928, type=float, help='kT/rc² = 12.928 mN.m')
parser.add_argument('--eps', default=1e-6, type=float, help='tolerance to declare coexistence')
parser.add_argument('--rho', default=3.0, type=float, help='baseline density, default 3.0')
parser.add_argument('-A', '--Alist', default='25,30,20', help='A11, A12, A22, default 25, 30, 20')
parser.add_argument('-g', '--guess', default='0.0001,0.99', help='initial guess x, default 0.0001, 0.99')
parser.add_argument('-v', '--verbose', action='count', default=0, help='increasing verbosity')
parser.add_argument('-s', '--show', action='store_true', help='plot the density profiles')
parser.add_argument('-t', '--thermodynamics', action='store_true', help='include thermodynamics in plot')
parser.add_argument('-x', '--exclude', action='store_true', help='phase equilibria only')
parser.add_argument('-o', '--output', help='output data for xmgrace, etc')
args = parser.parse_args()

def integral(f): # wrapper around numpy trapz 
    return np.trapz(f, dx=dz)

def kernel_convolve(ρ): # wrapper around numpy convolve
    return dz * np.convolve(ρ, kernel, mode='same')

def length(z): # return the length of a z domain
    return z[-1] - z[0]

def initial_density_profile(ρb): # step function
    ρ = np.zeros_like(z)
    ρ[z>0] = ρb[0]
    ρ[z<0] = ρb[1]
    ρ[z==0] = 0.5*(ρb[0] + ρb[1])
    return ρ

def clamp_density_profile(ρ, ρb): # outside domain
    ρ[rh_bulk] = ρb[0]
    ρ[lh_bulk] = ρb[1]
    return ρ

# Define a kernel K(z) in -1 < z < 1, and calculate the approximate
# value of pi/15 as its integral.

dz = args.dz

z = np.linspace(-1, 1, round(2/dz)+1, dtype=float)
kernel = π/12.0*(1-z)**3*(1+3*z)
kernel[z<0] = np.flip(kernel[z>0])
πby15 = integral(kernel)

if args.verbose:
    print(f'integrated kernel =\t{πby15}')
    print(f'exact result π/15 =\t{π/15}')

ρ0 = args.rho
A11, A12, A22 = eval(args.Alist) # returns a tuple

p0 = ρ0 + 1/2 * πby15 * args.Abase * ρ0**2 # reference pressure (NPT)

if args.verbose:
    print(f'ρ0, A11, A12, A22 = {ρ0} {A11} {A12} {A22}')
    print(f'pressure fixed at p0 = {p0}')

# First, solve for liquid-liquid coexistence in a DPD binary mixture
# described by mean-field free energy f = fid + fex where
#  fid = ρ1*(ln(ρ1/ρ0) - 1) + ρ2*(ln(ρ2/ρ0) - 1),
#  fex = π/30 (A11*ρ1² + 2*A12*ρ1*ρ2 + A22*ρ2²).
# The corresponding chemical potentials are (standard state is ρ0):
#  μ1 = ln(ρ1/ρ0) + π/15*(A11*ρ1 + A12*ρ2),
#  μ2 = ln(ρ2/ρ0) + π/15*(A12*ρ1 + A22*ρ2),
# and pressure p = ρ + π/30*(A11*ρ1**2 + 2*A12*ρ1*ρ2 + A22*ρ2**2),
# where ρ = ρ1 + ρ2.  Some tweaking of the initial guess may be needed
# to encourage the solver to find distinct coexisting state points.

# To impose NPT we solve a quadratic equation for ρ which is
#  π/30 [A11 x² + 2 A12 x(1-x) + A22 (1-x)²] ρ² + ρ - p0 = 0,
# on writing the pressure in terms of ρ1, ρ2 = xρ, (1-x)ρ.

# Use y = ln(x/(1-x)) which inverts to x = 1/(1+exp(-y)).  Note the
# function takes y as a 2-vector corresponding to the two state
# points, and returns the difference in the chemical potentials of the
# two components between two state points.

def fun(y, p0, A11, A12, A22):
    x = 1 / (1 + exp(-y)) # convert to a pair of mole fractions
    a = 1/2 * πby15 * (A11*x**2 + 2*A12*x*(1-x) + A22*(1-x)**2)
    ρ = (sqrt(1 + 4*a*p0) - 1) / (2*a) # impose NPT condition
    ρ1, ρ2 = x*ρ, (1-x)*ρ # the two mole fractions, at the two state points
    μ1 = ln(ρ1/ρ0) + πby15*(A11*ρ1 + A12*ρ2) # chemical potentials ..
    μ2 = ln(ρ2/ρ0) + πby15*(A12*ρ1 + A22*ρ2) # .. at the two state points
    return [μ1[1] - μ1[0], μ2[1] - μ2[0]]

x0 = np.array(eval(f'[{args.guess}]')) # mole fractions in the two phases (initial guess)
y0 = ln(x0 / (1-x0))
soln = root(fun, y0, args=(p0, A11, A12, A22))
yb = soln.x
xb = 1 / (1 + exp(-yb)) # final mole fractions in the two phases (a 2-vector)

if args.verbose > 2:
    print('Initial guess =', x0)
    print(soln)

if abs(xb[1]-xb[0]) < args.eps:
    raise ValueError(f'State points likely coalesced, x = {xb[0]}, {xb[1]}')

a = 1/2 * πby15 * (A11*xb**2 + 2*A12*xb*(1-xb) + A22*(1-xb)**2)
ρb = (sqrt(1 + 4*a*p0) - 1) / (2*a) # a 2-vector containing the total densities
ρ1b, ρ2b = xb*ρb, (1-xb)*ρb # these are 2-vectors containing the coexisting densities
μ1 = ln(ρ1b/ρ0) + πby15 * (A11*ρ1b + A12*ρ2b) # 2-vector, should be equal in coexistence
μ2 = ln(ρ2b/ρ0) + πby15 * (A12*ρ1b + A22*ρ2b) # -- ditto --
p = ρb + 1/2 * πby15 * (A11*ρ1b**2 + 2*A12*ρ1b*ρ2b + A22*ρ2b**2) # -- ditto --

# Consensus 'bulk' values used in the density profile calculations.

μ1b, μ2b, pb = map(np.mean, [μ1, μ2, p]) 

if args.verbose > 1 or args.exclude:
    print('Coexistence problem solution')
    for v in ['ρb', 'xb', 'ρ1b', 'ρ2b', 'μ1', 'μ2', 'p']:
        print(f'{v:>5} =\t', '\t'.join(map(str, eval(v))))
    for v in ['μ1b', 'μ2b', 'pb']:
        print(f'{v:>5} =\t', eval(v), '\t(consensus)')

if args.exclude:
    exit()

# Create an array z in [-zmax, zmax] and a computational domain
# [-zmax+1, zmax-1] within which the convolution operation is valid.

zmax = args.zmax
z = np.linspace(-zmax, zmax, round(2*zmax/dz)+1, dtype=float)
idx = np.round(z/dz).astype(int) # index with origin z = 0.0 --> 0
lh_bulk = (z < 1-zmax)
rh_bulk = (z > zmax-1)
domain = ~lh_bulk & ~rh_bulk

# Solve the following by Picard iteration :
#  ρ_i(z) = exp[ μ_i - ∑_j ∫ dz' ρ_j(z') U_ij(z-z') ]
# The integrals are evaluated as convolutions. Outside the domain
# where the convolution is valid, the density profiles are clamped to
# the bulk values.

α = args.alpha
tol = args.tolerance
max_iters = eval(args.max_iters.replace('^', '**'))

ρ1, ρ2 = map(initial_density_profile, [ρ1b, ρ2b])

for i in range(max_iters):
    ρ1_kern, ρ2_kern = map(kernel_convolve, [ρ1, ρ2])
    ρ1_new = clamp_density_profile(ρ0*exp(μ1b - A11*ρ1_kern - A12*ρ2_kern), ρ1b)
    ρ2_new = clamp_density_profile(ρ0*exp(μ2b - A12*ρ1_kern - A22*ρ2_kern), ρ2b)
    ρ1 = (1-α)*ρ1 + α*ρ1_new
    ρ2 = (1-α)*ρ2 + α*ρ2_new
    int_abs_Δρ1 = integral(np.abs(ρ1_new - ρ1))
    int_abs_Δρ2 = integral(np.abs(ρ2_new - ρ2))
    conv = int_abs_Δρ1 + int_abs_Δρ2
    if conv < tol: # early escape if converged
        break
else:
    raise ValueError(f'Picard failed to converge after {i} iterations, conv = {conv}')

if args.verbose:
    print(f'Picard converged after {i} iterations, conv = {conv}')

ρ = ρ1 + ρ2 # total density
x = ρ1 / ρ # local mole fraction

Δρ1 = ρ1 - initial_density_profile(ρ1b)
Δρ2 = ρ2 - initial_density_profile(ρ2b)

Γ1, Γ2 = [integral(Δρ[domain]) for Δρ in [Δρ1, Δρ2]] # surface excesses

# The wall tension is γ = Ω/A + 2 p Lz where Ω/A = ∫ dz ω and
#  ω = - ∑_i [ ρ_i(z) + 1/2 ∑_j ∫ dz' ρ_i(z) ρ_j(z') U_ij(z-z') ].
# The inner integrals are evaluated using convolution again

ρ1_kern, ρ2_kern = map(kernel_convolve, [ρ1, ρ2])
negωN1 = 1 + 1/2 * (A11*ρ1_kern + A12*ρ2_kern)
negωN2 = 1 + 1/2 * (A12*ρ1_kern + A22*ρ2_kern)
negω = ρ1*negωN1 + ρ2*negωN2
negΩbyA = integral(negω[domain])
twoLz = length(z[domain])
γ = pb*twoLz - negΩbyA

vs = ['twoLz', 'Γ1', 'Γ2', 'γ'] 
print(', '.join(vs), '(mN.m)\t', '\t'.join([str(eval(v)) for v in vs]), '\t', γ*args.ktbyrc2)

if args.output:

    import pandas as pd

    # Here we don't want to output every point with a discretisation
    # dz = 1e-3 or smaller, rather we downsample to a coarser grid.

    filtered = (np.mod(idx, round(args.gridz/dz)) == 0) # binary array
    grid = domain & filtered # values to write out
    data = np.array([z[grid], ρ1[grid], ρ2[grid], ρ[grid], x[grid], negωN1[grid], negωN2[grid], negω[grid]])
    df = pd.DataFrame(data.transpose(), columns=['z', 'ρ1', 'ρ2', 'ρ', 'x', 'negωN1', 'negωN2', 'negω'])
    with open(args.output, 'w') as f:
        print(df_to_agr(df), file=f)
    print('Data:', ', '.join(df_header(df)), 'written to', args.output)
 
elif args.show:

    import matplotlib.pyplot as plt

    region = ~(z<-args.zcut) & ~(z>args.zcut)

    plt.plot(z[region], ρ1[region])
    plt.plot(z[region], ρ2[region])
    #plt.plot(z[region], Δρ1[region])
    #plt.plot(z[region], Δρ2[region])
    if args.thermodynamics:
        plt.plot(z[region], negωN1[region])
        plt.plot(z[region], negωN2[region])
        plt.plot(z[region], pb - negω[region])
    #plt.plot(z[region], ρ[region])
    #plt.plot(z[region], x[region])

    plt.show()
