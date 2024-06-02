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

# For a matched (Awall = Abulk) continuum wall, ρ(z) = ρb for z > 0,
# so that the surface excess vanishes.  However γ = π A ρb²/240 ; for
# standard water (A = 25, ρb = 3), this is γ = 15π/16 ≈ 2.94524.

# For benchmarking, this should be the output for a standard wall with
# the default arguments:
#  Abs deviation in bulk = 0.0487065
#  Surface excess per unit area Γ/A = 0.19294
#  Wall tension γ = 1.63487 = 21.1355 mN.m
# and the same for a continuum wall (note the negative wall tension):
#  Abs deviation in bulk = 0.0348605
#  Surface excess per unit area Γ/A = 0.447014
#  Wall tension γ = -1.62926 = -21.0631 mN.m

import wallDFT
from wallDFT import wall_args, df_header, df_to_agr

eparser = wallDFT.ExtendedArgumentParser(description='DPD wall profile one off calculator')
eparser.add_argument('--zcut', default=4.0, type=float, help='cut-off in z, default 4.0')
eparser.add_argument('--gridz', default=0.02, type=float, help='filter spacing in z, default 0.02')
eparser.add_argument('-s', '--show', action='store_true', help='plot the density profile')
eparser.add_argument('-o', '--output', help='output data for xmgrace, etc')
args = eparser.parse_args()

wall = wallDFT.Wall(**wall_args(args))

print(wall.about)

Awall = eval(args.Awall)
rhow = eval(args.rhow)
Abulk = eval(args.Abulk)
rhob = eval(args.rhob)

wall.solve_and_print(Awall, rhow, Abulk, rhob, args)

if args.output:

    import numpy as np
    import pandas as pd

    ρ = wall.density_profile()

    # Here we don't want to output every point with a discretisation
    # dz = 1e-3 or smaller, rather we downsample to a coarser grid.

    filtered = (np.mod(wall.idx, round(args.gridz/wall.dz)) == 0) # binary array
    plot_region = ~(wall.z < 0) & ~(wall.z > args.zcut) # another binary array
    grid = plot_region & filtered # grid of positions to write out
    data = np.array([wall.z[grid], ρ[grid], wall.uwall[grid]]).transpose()
    df = pd.DataFrame(data, columns=['z', 'ρ', 'uwall'])
    with open(args.output, 'w') as f:
        print(df_to_agr(df), file=f)
    print('Data:', ', '.join(df_header(df)), 'written to', args.output)
 
elif args.show:

    import matplotlib.pyplot as plt

    ρ = wall.density_profile()

    plot_region = ~(wall.z > args.zcut) # another binary array

    plt.plot(wall.z[plot_region], ρ[plot_region])
    plt.plot(wall.z[plot_region], wall.uwall[plot_region])
    plt.show()
