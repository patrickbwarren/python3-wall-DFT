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

eparser = wall_dft.ExtendedArgumentParser(description='DPD wall profile one off calculator')
eparser.add_argument('--zcut', default=4.0, type=float, help='cut-off in z, default 4.0')
eparser.add_argument('--gridz', default=0.02, type=float, help='filter spacing in z, default 0.02')
eparser.add_argument('--vanilla', action='store_true', help='use vanilla wall model')
eparser.add_argument('--conversion-factor', default=12.928, type=float, help='conversion factor kT/rc² = 12.928 mN.m')
eparser.add_argument('-s', '--show', action='store_true', help='plot the density profile')
eparser.add_argument('-o', '--output', help='output plot to, eg, pdf')
args = eparser.parse_args()

max_iters = eval(args.max_iters.replace('^', '**'))

wall = wall_dft.Wall(dz=args.dz, zmax=args.zmax)

print(wall.about)

rhob = eval(args.rhob)
Abulk = eval(args.Abulk)
Awall = eval(args.Awall)

wall.continuum_wall(Awall*rhob) if args.continuum else wall.standard_wall(Awall)
print(wall.model)

iters, convergence = wall.solve(rhob, Abulk, max_iters=max_iters, alpha=args.alpha, tol=args.tolerance)

ρ = wall.density_profile()
Γ = wall.surface_excess()
w = wall.abs_deviation()
γ, ωb, Lz = wall.wall_tension()
p_mf = rhob + π/30 * Abulk * rhob**2

print('Converged after %i iterations, ∫dz|ΔΔρ| = %g' % (iters, convergence))
print('Abulk, ρb = %g, %g' % (Abulk, rhob))
print('Surface excess per unit area Γ/A = %g' % Γ)
print('Bulk grand potential ωb = %g' % ωb)
print('Bulk mean field pressure, p = %g' % p_mf)
print('Domain size Lz = %g' % Lz)
print('Surface tension γ = %g ' % γ)
print('Abs deviation = %g' % w)

print('Surface tension γ = %g mN.m' % (γ * args.conversion_factor))

if args.output:

    import numpy as np
    import pandas as pd

    # Here we don't want to output every point with a discretisation
    # dz = 1e-3 or smaller, rather we downsample to a coarser grid.

    filtered = (np.mod(wall.idx, round(args.gridz/wall.dz)) == 0) # binary array
    plot_region = ~(wall.z < 0) & ~(wall.z > args.zcut) # another binary array
    grid = plot_region & filtered # values to write out
    df = pd.DataFrame(np.array([wall.z[grid], ρ[grid], wall.uwall[grid]]).transpose(),
                      columns=['z', 'ρ', 'uwall'])
    df.to_csv(args.output, sep='\t', header=False, index=False, float_format='%g')
    print(', '.join([f'{s}({i+1})' for i, s in enumerate(df.columns)]), 'written to', args.output)
 
elif args.show:

    import matplotlib.pyplot as plt

    plot_region = ~(wall.z > args.zcut) # another binary array

    plt.plot(wall.z[plot_region], ρ[plot_region])
    plt.plot(wall.z[plot_region], wall.uwall[plot_region])
    plt.show()
