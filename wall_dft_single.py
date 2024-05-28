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

import argparse
import wall_dft

parser = argparse.ArgumentParser(description='DPD wall profile one off calculator')
wall_dft.common_arguments(parser)
parser.add_argument('--zcut', default=4.0, type=float, help='cut-off in z, default 4.0')
parser.add_argument('--gridz', default=0.02, type=float, help='filter spacing in z, default 0.02')
parser.add_argument('--show', action='store_true', help='plot the density profile')
parser.add_argument('--vanilla', action='store_true', help='use vanilla wall model')
parser.add_argument('--conversion-factor', default=12.928, type=float, help='conversion factor kT/rc² = 12.928 mN.m')
parser.add_argument('-o', '--output', help='output plot to, eg, pdf')
args = parser.parse_args()

max_iters = eval(args.max_iters.replace('^', '**'))

wall = wall_dft.Wall(dz=args.dz, zmax=args.zmax)

print(wall.about)

wall.rhob = eval(args.rhob)
wall.Abulk = eval(args.Abulk)
wall.Awall = eval(args.Awall)
wall.model = 'half space' if args.half_space else 'vanilla'

wall.solve(max_iters=max_iters, alpha=args.alpha, tol=args.tolerance)

print(wall.properties)
print('Surface tension γ = %g mN.m' % (wall.gamma * args.conversion_factor))

if args.output:

    import numpy as np
    import pandas as pd

    # Here we don't want to output every point with a discretisation
    # dz = 1e-3 or smaller, rather we downsample to a coarser grid.

    filtered = (np.mod(wall.idx, round(args.gridz/wall.dz)) == 0) # binary array
    plot_region = ~(wall.z < 0) & ~(wall.z > args.zcut) # another binary array
    grid = plot_region & filtered # values to write out
    df = pd.DataFrame(np.array([wall.z[grid], wall.ρ[grid], wall.uwall[grid]]).transpose(),
                      columns=['z', 'ρ', 'uwall'])
    df.to_csv(args.output, sep='\t', header=False, index=False, float_format='%g')
    print(', '.join([f'{s}({i+1})' for i, s in enumerate(df.columns)]), 'written to', args.output)
 
elif args.show:

    import matplotlib.pyplot as plt

    plot_region = ~(wall.z > args.zcut) # another binary array

    plt.plot(wall.z[plot_region], wall.ρ[plot_region])
    plt.plot(wall.z[plot_region], wall.uwall[plot_region])
    plt.show()
