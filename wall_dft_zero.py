#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Code to calculate surface density profiles, surface excess, and
# surface tension, for DPD wall models with a simple DPD fluid.

# This code is part of DPDWall-DFT, copyright (c) 2024 Patrick B Warren (STFC).
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

import argparse
import wall_dft
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar, minimize_scalar

parser = argparse.ArgumentParser(description='DFT wall property zero calculator')
wall_dft.common_arguments(parser)
parser.awall.default = '0,40'
parser.awall.help='wall repulsion amplitude bracket, default 0,40'
parser.add_argument('--gamma', action='store_true', help='zero surface tension, otherwise surface excess')
parser.add_argument('--wobble', action='store_true', help='minimum wobbliness')
args = parser.parse_args()

max_iters = eval(args.max_iters.replace('^', '**'))

Alo, Ahi = eval(args.Awall)

wall = wall_dft.Wall(dz=args.dz, zmax=args.zmax)

print(wall.about)

wall.rhob = eval(args.rhob)
wall.Abulk = eval(args.Abulk)
wall.model = 'half space' if args.half_space else 'vanilla'

def surface_excess(Awall, gamma):
    wall.Awall = Awall
    wall.solve(max_iters=max_iters, alpha=args.alpha, tol=args.tolerance)
    return wall.gamma if gamma else wall.surface_excess

def wobbliness(Awall):
    wall.Awall = Awall
    wall.solve(max_iters=max_iters, alpha=args.alpha, tol=args.tolerance)
    return wall.wobble

if args.wobble:

    sol = minimize_scalar(wobbliness, bracket=(1, 5), bounds=(0, 50))
    print('Solution minimising wobbliness')
    print(sol)

else:

    sol = root_scalar(surface_excess, bracket=(Alo,Ahi), method='brentq', args=(args.gamma,))
    print('Solution for vanishing', 'surface tension' if args.gamma else 'surface excess')
    print(sol)

wall.Awall = sol.x if args.wobble else sol.root
wall.solve(max_iters=max_iters, alpha=args.alpha, tol=args.tolerance)

print(wall.properties)
