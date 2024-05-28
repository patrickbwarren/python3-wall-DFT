#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Code to calculate surface density profiles, surface excess, and
# surface tension, for DPD wall models with a simple DPD fluid.

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

import wall_dft
from numpy import pi as π
from scipy.optimize import root_scalar as find_zero

eparser = wall_dft.ExtendedArgumentParser(description='DPD wall profile zero calculator')
eparser.awall.default = '0,40'
eparser.awall.help='wall repulsion amplitude bracket, default ' + eparser.awall.default
eparser.add_argument('--ktbyrc2', default=12.928, type=float, help='kT/rc² = 12.928 mN.m')
eparser.add_argument('-g', '--gamma', action='store_true', help='search for zero surface tension, rather than surface excess')
args = eparser.parse_args()

max_iters = eval(args.max_iters.replace('^', '**'))

Alo, Ahi = eval(args.Awall)
rhob = eval(args.rhob)
Abulk = eval(args.Abulk)

wall = wall_dft.Wall(dz=args.dz, zmax=args.zmax)

if args.verbose:
    print(wall.about)

def func(Awall):
    wall.continuum_wall(Awall*rhob) if args.continuum else wall.standard_wall(Awall)
    iters, conv = wall.solve(rhob, Abulk, max_iters=max_iters, alpha=args.alpha, tol=args.tolerance)
    Γ = wall.surface_excess()
    w = wall.abs_deviation()
    γ, _, _ = wall.wall_tension()
    if args.verbose > 1:
        print('%g\t%g\t%g\t%g\t%g\t%i' % (Awall, Γ, γ, w, conv, iters))
    return γ if args.gamma else Γ

sol = find_zero(func, bracket=(Alo, Ahi), method='brentq')

print('Solution for vanishing surface', 'tension' if args.gamma else 'excess')

if args.verbose:
    print('     converged:', sol.converged)
    print('function_calls:', sol.function_calls)
    print('    iterations:', sol.iterations)
    print('          root:', sol.root)

Awall = sol.root

wall.continuum_wall(Awall*rhob) if args.continuum else wall.standard_wall(Awall)
print(wall.model)

iters, conv = wall.solve(rhob, Abulk, max_iters=max_iters, alpha=args.alpha, tol=args.tolerance)

ρ = wall.density_profile()
Γ = wall.surface_excess()
w = wall.abs_deviation()
γ, ωb, Lz = wall.wall_tension()
p_mf = rhob + π/30 * Abulk * rhob**2

print('Converged after %i iterations, ∫dz|ΔΔρ| = %g' % (iters, conv))
print('Awall, Abulk, ρb = %g, %g, %g' % (Awall, Abulk, rhob))
print('Surface excess per unit area Γ/A = %g' % Γ)
print('Bulk grand potential ωb = %g' % ωb)
print('Bulk mean field pressure, p = %g' % p_mf)
print('Domain size Lz = %g' % Lz)
print('Surface tension γ = %g ' % γ)
print('Abs deviation = %g' % w)

print('Surface tension γ = %g mN.m' % (γ * args.ktbyrc2))
