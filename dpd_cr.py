#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Modified from https://github.com/patrickbwarren/python3-HNC-solver

# This program is part of pyHNC, copyright (c) 2023 Patrick B Warren (STFC).
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

import pyHNC
import argparse
import numpy as np
from numpy import pi as π
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(description='DPD HNC calculator')
parser.add_argument('-A', '--A', default='25.0', help='repulsion amplitude, default 25.0')
parser.add_argument('-r', '--rho', default='3.0', help='density, default 3.0')
parser.add_argument('--rcut', default=3.0, type=float, help='maximum in r for plotting, default 3.0')
parser.add_argument('-s', '--show', action='store_true', help='show results')
parser.add_argument('-o', '--output', help='write functions to a file')
parser.add_argument('-v', '--verbose', action='count', help='more details (repeat as required)')
args = parser.parse_args()

A, ρ = eval(args.A), eval(args.rho)

#grid = pyHNC.Grid(ng=2**14, deltar=1e-2) # make the initial working grid
grid = pyHNC.Grid(ng=2**18, deltar=1e-3) # make the initial working grid
r, dr = grid.r, grid.deltar # extract the co-ordinate arrays for use below

if args.verbose:
    print(grid.details)

# The DPD potential, its derivative

φ = A * pyHNC.truncate_to_zero(1/2*(1-r)**2, r, 1) # the DPD potential
f = A * pyHNC.truncate_to_zero((1-r), r, 1) # the force f = -dφ/dr

dz = dr
z = np.linspace(-1.0, 1.0, round(2.0/dz)+1, dtype=float)
U = A*π/12.0*(1-z)**3*(1+3*z)
U[z<0] = np.flip(U[z>0])

# Solve the HNC problem (need to include 'A' in the supplied potential)

solver = pyHNC.PicardHNC(grid)
soln = solver.solve(φ, ρ, monitor=args.verbose)
h, c = soln.hr, soln.cr
g = 1 + h

if args.verbose:
    print(solver.details)

# Fit -c(r) to a (1-r)² (1 + 2 r + b r² + c r³) for r < 1

def poly(r, a, b, c): # satisfies p[0] = a, p'[0] = 0, p[1] = p'[1] = 0
    return a * (1-r)**2 * (1 + 2*r + b*r**2 + c*r**3)

aa, bb, cc = curve_fit(poly, r[r<1], -c[r<1])[0] # fit p(r) to -c(r) on 0 < r < 1

p = pyHNC.truncate_to_zero(poly(r, aa, bb, cc), r, 1) # the fitted polynomial 

def r_poly_int(z, a, b, c): # being ∫_z dr r p(r), evaluated analytically and valid on 0 < z < 1
    return a*(1-z)**3/420 * (63+4*c+21*z*(9+8*z) +  4*c*z*(3+6*z+10*z**2+15*z**3) + 7*b*(1+3*z+6*z**2+10*z**3))

print()
print('Fitted polynomial coefficients in p(r) = -c(r): a, b, c = %0.6f %0.6f %0.6f\n' % (aa, bb, cc))

# For reference (A, ρ = 25, 3) the fit values are 6.615111 0.279768 -1.935711

aa, bb, cc = 6.615111, 0.279768, -1.935711 # use the 6-fig accuracy values for A, ρ = 25, 3

# The following evaluates ∫_z dx f(x), such that the zeroth array
# entry matches np.trapz of the same.

def partial_integral(f, dx=1):
    integrand = 0.5*(f[1:] + f[:-1])
    summand = dx * np.pad(integrand, (1, 0))
    return np.flip(np.flip(summand).cumsum())

Ua = 2*π*partial_integral(r*φ, dx=dr)
Kc = 2*π*partial_integral(-r*c, dx=dr)

Kp = 2*π*pyHNC.truncate_to_zero(r_poly_int(r, aa, bb, cc), r, 1)
Kp0 = aa*π/210 * (63 + 7*bb + 4*cc) # being ∫_0 dr r p(r), evaluated analytically

print('Partial integral limiting values')
print()
print('          U(z=0) =\t', U[z==0][0], '\tpotential φ(r) integrals')
print('  2π ∫ dr r φ(r) =\t', 2*π*np.trapz(r*φ, dx=dr), '\t(should be close to the above)')
print('           Ua(0) =\t', Ua[0], '\t(should be the same as above)')
print()
print('- 2π ∫ dr r c(r) =\t', 2*π*np.trapz(-r*c, dx=dr), '\tdirect correlation fn c(r) integrals')
print('           Kc(0) =\t', Kc[0], '\t(should be the same as above)')
print('           Kp(0) =\t', Kp0, '\t(should be fairly close to the above)\n')

# We want to compare the various approximations here.
# The virial pressure is p = ρ - 2πρ²/3 ∫ dr r³ φ' g.

# The compressibility dp/dρ = 1 - ρ ∫ d³r c(r) can also be
# expressed using the Kc function defined above,
#  ∫ dr 4πr² c(r) = ∫ dz Kc(z) from z = 0 to ∞.

Uanorm = np.trapz(Ua, dx=dr)
Kcnorm = np.trapz(Kc, dx=dr)

Kpnorm = aa*π/420 * (56 + 8*bb + 5*cc) # being ∫ dz Kp(z), evaluated analytically

# For standard DPD at A = 25 and ρ = 3, we have the following table

#           ∆t = 0.02   ∆t = 0.01   Monte-Carlo  HNC   deviation
# pressure  23.73±0.02  23.69±0.02  23.65±0.02   23.564  (0.4%)
# energy    13.66±0.02  13.64±0.02  13.63±0.02   13.762  (1.0%)
# mu^ex     12.14±0.02  12.16±0.02  12.25±0.10   12.170  (0.7%)

pMC, err = 23.65, 0.02

print('     HNC compressibility =\t', 1-4*π*ρ*np.trapz(r**2*c, dx=dr))
print('    Monte-Carlo pressure =\t', pMC, '±', err)
print('     HNC virial pressure =\t', ρ + 2*π/3*ρ**2*np.trapz(r**3*f*g, dx=dr))
print('HNC pressure using -c(r) =\t', ρ - 2*π*ρ**2*np.trapz(r**2*c, dx=dr))
print(' HNC pressure using φ(r) =\t', ρ + 2*π*ρ**2*np.trapz(r**2*φ, dx=dr))
print(' MF pressure ρ + πAρ²/30 =\t', ρ + π*A*ρ**2/30)
print()

print('Coefficients α in p = ρ + αAρ²')
print(f'   Monte-Carlo coeff =\t{(pMC-3)/(25*3**2):0.5g} ± {err/(25*3**2):0.5f}')
print(f'HNC virial EOS coeff =\t{2*π/3 * np.trapz(r**3*f*g, dx=dr)/A}')
print(f'  HNC comp EOS coeff =\t{2*π*np.trapz(-r**2*c, dx=dr)/A}')
print(f'  Alt comp EOS coeff =\t{Kcnorm/A}\t(should be the same as above; from c(r))')
print(f'  Alt comp EOS coeff =\t{Kpnorm/A}\t(should be fairly close to the above; from p(r))\n')

print(f'MF EOS coeff A*π/30 =\t{A*π/30}\t(all these should be the same)')
print(f'   Alt MF EOS coeff =\t{2*π*np.trapz(r**2*φ, dx=dr)}')
print(f'   Alt MF EOS coeff =\t{Uanorm}\t<-- use this one !')
print(f'   Alt MF EOS coeff =\t{0.5*np.trapz(U, dx=dr)}\n')

# The factor here multiplicatively renormalises A in a mean field DFT
# approach, which in the present context of a bulk system is the same
# as the mean field van der Waals EOS.

renorm = 0.8764 # empirically determined as MC excess pressure / MF excess pressure

print(f' pMC(ex) / pMF(ex) =\t{(pMC-3)/(A*π*ρ**2/30)}')
print(f'pHNC(ex) / pMF(ex) =\t{20*np.trapz(r**3*f*g, dx=dr)/A}')
print(f'     renorm factor =\t{renorm}\t<-- using this value !\n')

print('Coefficient of ρ² in p = ρ + αAρ²')
print(f'   Monte-Carlo coeff =\t {(pMC-3)/3**2:0.3g} ± {err/3**2:0.3f}')
print(f'  MF EOS coeff Aπ/30 =\t{A*π/30}')
print(f'    Alt MF EOS coeff =\t{np.trapz(Ua, dx=dr)}\t(should be the same as above)')
print(f' renorm MF EOS coeff =\t{renorm*np.trapz(Ua, dx=dr)}')
print(f'HNC virial EOS coeff =\t{2*π/3*np.trapz(r**3*f*g, dx=dr)}')
print(f'  HNC comp EOS coeff = \t{2*π*np.trapz(-r**2*c, dx=dr)}')
print(f'  Alt comp EOS coeff = \t{Kcnorm}\t(should be the same as above; from c(r))')
print(f'  Alt comp EOS coeff = \t{Kpnorm}\t(should be fairly close to the above; from p(r))\n')

print('Actual pressures')
print(f'            Monte-Carlo =\t{pMC:0.2f} ± {err:0.2f}')
print(f'            MF pressure =\t{ρ+A*π*ρ**2/30}')
print(f'        Alt MF pressure =\t{ρ+ρ**2*Uanorm}\t(should be the same as above)')
print(f'   pressure using -c(r) =\t{ρ+ρ**2*Kcnorm}')
print(f'    HNC virial pressure =\t{ρ+2*π/3*ρ**2*np.trapz(r**3*f*g, dx=dr)}\n')
print(f'     renorm MF pressure =\t{ρ+renorm*A*π*ρ**2/30}')
print(f' alt renorm MF pressure =\t{ρ+renorm*ρ**2*Uanorm}\t(should be the same as above)')

if args.show:

    import matplotlib.pyplot as plt

    rcut = r < args.rcut
    zcut = ~(z < 0) & (z < args.rcut)

    plt.figure(1)
    plt.plot(r[rcut], φ[rcut], label='φ(r)')
    plt.plot(r[rcut], p[rcut], label='p(r)')
    plt.plot(r[rcut], -c[rcut], label='-c(r)')
    plt.plot(r[rcut], g[rcut], label='g(r)')
    plt.legend() ; plt.xlabel('r')
    plt.show()

    plt.figure(2)
    plt.plot(r[rcut], r[rcut]*φ[rcut], label='r φ(r)')
    plt.plot(r[rcut], r[rcut]*p[rcut], label='r p(r)')
    plt.plot(r[rcut], -r[rcut]*c[rcut], label='-r c(r)')
    plt.legend() ; plt.xlabel('r')
    plt.show()

    plt.figure(3)
    plt.plot(z[zcut], U[zcut], label='U(z)')
    # plt.plot(r[rcut], Ua[rcut], label='Ua(r)')
    plt.plot(r[rcut], Kp[rcut], label='K[p](z)')
    plt.plot(r[rcut], Kc[rcut], label='K[c](z)')
    plt.legend() ; plt.xlabel('z') # ; plt.ylabel('vanilla')
    plt.show()

    plt.figure(4)
    plt.plot(z[zcut], U[zcut]*30/(π*A), label='U(z) / norm')
    # plt.plot(r[rcut], Ua[rcut]/Uanorm, label='Ua(z) / norm')
    plt.plot(r[rcut], Kp[rcut]/Kpnorm, label='K[p](z) / norm')
    plt.plot(r[rcut], Kc[rcut]/Kcnorm, label='K[c](z) / norm')
    plt.legend() ; plt.xlabel('z') #; plt.ylabel('normed by integral')
    plt.show()

    plt.figure(5)
    plt.plot(z[zcut], U[zcut]*12/(π*A), label='U(z) / U(0)')
    # plt.plot(r[rcut], Ua[rcut]/Ua[0], label='Ua(z)')
    plt.plot(r[rcut], Kc[rcut]/Kp0, label='K[p](z) / K[p](0)')
    plt.plot(r[rcut], Kc[rcut]/Kc[0], label='K[c](z) / K[c](0)')
    plt.legend() ; plt.xlabel('z') #; plt.ylabel('normed by f[0]')
    plt.show()

if args.output:

    # this needs fixing up to output the various functions above
    
    import pandas as pd

    rcut = r < args.rcut

    df = pd.DataFrame({'r': r[rcut], 'g': g[rcut]})
    df_agr = pyHNC.df_to_agr(df)

    with open(args.output, 'w') as f:
        print(f'# DPD with A = {A:g}, ρ = {ρ:g}, HNC closure\n' + df_agr, file=f)

    print('Written (r, g) to', args.output)
