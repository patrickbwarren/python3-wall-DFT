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
from pyHNC import Grid, PicardHNC, truncate_to_zero

parser = argparse.ArgumentParser(description='DPD HNC calculator')
parser.add_argument('-A', '--A', default='25.0', help='repulsion amplitude, default 25.0')
parser.add_argument('-r', '--rho', default='3.0', help='density, default 3.0')
parser.add_argument('--rcut', default=3.0, type=float, help='maximum in r for plotting, default 3.0')
parser.add_argument('-s', '--show', action='store_true', help='show results')
parser.add_argument('-o', '--output', help='write functions to a file')
parser.add_argument('-v', '--verbose', action='count', help='more details (repeat as required)')
args = parser.parse_args()

A, ρ = eval(args.A), eval(args.rho)

#grid = Grid(ng=2**14, deltar=1e-2) # make the initial working grid
grid = Grid(ng=2**18, deltar=1e-3) # make the initial working grid
r, dr = grid.r, grid.deltar # extract the co-ordinate arrays for use below

if args.verbose:
    print(grid.details)

# The DPD potential, its derivative, and the 1d function, normed by A

φ = truncate_to_zero(1/2*(1-r)**2, r, 1) # the DPD potential
f = truncate_to_zero((1-r), r, 1) # the force f = -dφ/dr

dz = dr
z = np.linspace(-1.0, 1.0, round(2.0/dz)+1, dtype=float)
U = π/12.0*(1-z)**3*(1+3*z)
U[z<0] = np.flip(U[z>0])

# Solve the HNC problem (need to include 'A' in the supplied potential)

solver = PicardHNC(grid)
soln = solver.solve(A*φ, ρ, monitor=args.verbose)
h, c = soln.hr, soln.cr
g = 1 + h

if args.verbose:
    print(solver.details)

# The following evaluates ∫_z dx x f(x), such that the zeroth array
# entry matches np.trapz of the same.

def partial_integral(f, dx=1):
    integrand = 0.5*(f[1:] + f[:-1])
    summand = dx * np.pad(integrand, (1, 0))
    return np.flip(np.flip(summand).cumsum())

Ua = 2*π*partial_integral(r*φ, dx=dr)
Uc = 2*π*partial_integral(-r*c, dx=dr)

print('\nPartial integral limiting values\n')

print('U(z=0) =          \t', U[z==0][0], ' (should be close to the following)')
print('2π ∫ dr r φ(r) =\t', 2*π*np.trapz(r*φ, dx=dr))
print('Ua(0) =            \t', Ua[0], ' (should be the same as above)\n')
print('- 2π/A ∫ dr r c(r) =\t', 2*π*np.trapz(-r*c, dx=dr)/A)
print('Uc(0)/A =            \t', Uc[0]/A, ' (should be the same as above)\n')

# We want to compare the various approximations here.
# The virial pressure is p = ρ - 2πρ²/3 ∫ dr r³ φ' g.

# The compressibility dp/dρ = 1 - ρ ∫ d³r c(r) can also be
# expressed using the Uc function defined above,
#  ∫ dr 4πr² c(r) = ∫ dz Uc(z) from z = 0 to ∞.

Uanorm = np.trapz(Ua, dx=dr)
Ucnorm = np.trapz(Uc, dx=dr)

# For standard DPD at A = 25 and ρ = 3, we have the following table

#           ∆t = 0.02   ∆t = 0.01   Monte-Carlo  HNC   deviation
# pressure  23.73±0.02  23.69±0.02  23.65±0.02   23.564  (0.4%)
# energy    13.66±0.02  13.64±0.02  13.63±0.02   13.762  (1.0%)
# mu^ex     12.14±0.02  12.16±0.02  12.25±0.10   12.170  (0.7%)

pMC, err = 23.65, 0.02

print('\nCoefficients α in p = ρ + αAρ²\n')
print(f'Monte-Carlo coeff =\t{(pMC-3)/(25*3**2):0.5g} ± {err/(25*3**2):0.5f}')
print(f'HNC virial EOS coeff =\t{2*π/3 * np.trapz(r**3*f*g, dx=dr)}')
print(f'HNC compr EOS coeff = \t{2*π*np.trapz(-r**2*c, dx=dr)/A}')
print(f'Alt compr EOS coeff = \t{np.trapz(Uc, dx=dr)/A}\n')

print(f'MF EOS coeff π/30 =\t{π/30}')
print(f'Alt MF EOS coeff = \t{2*π*np.trapz(r**2*φ, dx=dr)}')
print(f'Alt MF EOS coeff = \t{np.trapz(Ua, dx=dr)} <-- use this one !')
print(f'Alt MF EOS coeff = \t{0.5*np.trapz(U, dx=dr)}\n')

# The factor here multiplicatively renormalises A in a mean field DFT
# approach, which in the present context of a bulk system is the same
# as the mean field van der Waals EOS.

renorm = Uc[0]/(A*Ua[0]) # note that 'A' is included in Uc by use of 'c'

print(f'Renormalisation Ucnorm / Uanorm =\t{Ucnorm/(A*Uanorm)}')
print(f'Renormalisation Uc[0] / Ua[0] =\t{renorm} <-- use this one !\n')

print('Coefficient of ρ² in p = ρ + αAρ²\n')
print(f'Monte-Carlo coeff =\t {(pMC-3)/3**2:0.3g} ± {err/3**2:0.3f}')
print(f'MF EOS coeff Aπ/30 =\t{A*π/30}')
print(f'Alt MF EOS coeff =\t{A*np.trapz(Ua, dx=dr)}')
print(f'renorm MF EOS coeff =\t{renorm*A*np.trapz(Ua, dx=dr)}')
print(f'HNC virial EOS coeff =\t{2*π/3*A*np.trapz(r**3*f*g, dx=dr)}')

print('Actual pressures\n')
print(f'Monte-Carlo =     \t{pMC:0.2f} ± {err:0.2f}')
print(f'MF EOS coeff Aπ/30 =\t{ρ+A*π*ρ**2/30}')
print(f'Alt MF EOS coeff =\t{ρ+A*ρ**2*np.trapz(Ua, dx=dr)}')
print(f'renorm MF EOS coeff =\t{ρ+renorm*A*ρ**2*np.trapz(Ua, dx=dr)}')
print(f'HNC virial EOS coeff =\t{ρ+2*π/3*A*ρ**2*np.trapz(r**3*f*g, dx=dr)}')

if args.show:

    import matplotlib.pyplot as plt

    rcut = r < args.rcut
    zcut = ~(z < 0) & (z < args.rcut)

#    plt.figure(1)
#    plt.plot(r[rcut], r[rcut]*φ[rcut])
#    plt.plot(r[rcut], -r[rcut]*c[rcut]/A)

    plt.figure(1)
    plt.plot(z[zcut], U[zcut])
    plt.plot(r[rcut], Ua[rcut])
    plt.plot(r[rcut], Uc[rcut]/A)

    plt.figure(2)
    plt.plot(z[zcut], U[zcut]*30/π)
    plt.plot(r[rcut], Ua[rcut]/Uanorm)
    plt.plot(r[rcut], Uc[rcut]/Ucnorm)

    plt.figure(3)
    plt.plot(z[zcut], U[zcut]*12/π)
    plt.plot(r[rcut], Ua[rcut]/Ua[0])
    plt.plot(r[rcut], Uc[rcut]/Uc[0])

    plt.show()

if args.output:

    # this needs fixing up to output the various functions above
    
    import pandas as pd

    rcut = r < args.rrcut

    df = pd.DataFrame({'r': r[rcut], 'g': g[rcut]})
    df_agr = pyHNC.df_to_agr(df)

    with open(args.output, 'w') as f:
        print(f'# DPD with A = {A:g}, ρ = {ρ:g}, HNC closure\n' + df_agr, file=f)

    print('Written (r, g) to', args.output)
