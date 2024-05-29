#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Code to calculate surface density profiles, surface excess, and
# wall tension, for DPD wall models with a simple DPD fluid.

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

import wallDFT
from numpy import pi as π
from scipy.optimize import root_scalar as find_zero
from wallDFT import wall_args, solve_args

eparser = wallDFT.ExtendedArgumentParser(description='DPD wall profile zero calculator')
eparser.awall.default = '0,40'
eparser.awall.help='wall repulsion amplitude bracket, default ' + eparser.awall.default
eparser.add_argument('-g', '--gamma', action='store_true', help='zero wall tension, rather than zero surface excess')
args = eparser.parse_args()

Alo, Ahi = eval(args.Awall)
Abulk = eval(args.Abulk)
rhob = eval(args.rhob)
rhow = eval(args.rhow)

wall = wallDFT.Wall(**wall_args(args))

if args.verbose:
    print(wall.about)

def func(Awall):
    wall.continuum_wall(Awall*rhow) if args.continuum else wall.standard_wall(Awall)
    iters, conv = wall.solve(rhob, Abulk, **solve_args(args))
    Γ = wall.surface_excess()
    w = wall.abs_deviation()
    γ, _, _ = wall.wall_tension()
    if args.verbose > 1:
        print('%g\t%g\t%g\t%g\t%g\t%i' % (Awall, Γ, γ, w, conv, iters))
    return γ if args.gamma else Γ

sol = find_zero(func, bracket=(Alo, Ahi), method='brentq')

print('Solution for vanishing', 'wall tension' if args.gamma else 'surface excess')

if args.verbose:
    print('     converged:', sol.converged)
    print('function_calls:', sol.function_calls)
    print('    iterations:', sol.iterations)
    print('          root:', sol.root)

Awall = sol.root

wall.solve_and_print(Awall, rhow, Abulk, rhob, args)
