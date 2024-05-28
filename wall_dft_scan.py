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
import numpy as np
import pandas as pd

eparser = wall_dft.ExtendedArgumentParser(description='DFT wall property table calculator')
eparser.awall.default = '0,40,5'
eparser.awall.help='wall repulsion amplitudes, default 0,40,5'
eparser.add_argument('--conversion-factor', default=12.928, type=float, help='conversion factor kT/rc² = 12.928 mN.m')
eparser.add_argument('-o', '--output', default=None, help='output data to, eg, .dat')
args = eparser.parse_args()

max_iters = eval(args.max_iters.replace('^', '**'))

Alo, Ahi, Astep = eval(f'({args.Awall})')
Awalls = np.linspace(Alo, Ahi, round((Ahi-Alo)/Astep)+1, dtype=float)

wall = wall_dft.Wall(dz=args.dz, zmax=args.zmax)

print(wall.about)

rhob = eval(args.rhob)
Abulk = eval(args.Abulk)

results = []
for i, Awall in enumerate(Awalls):
    wall.continuum_wall(Awall*rhob) if args.continuum else wall.standard_wall(Awall)
    iters, convergence = wall.solve(rhob, Abulk, max_iters=max_iters, alpha=args.alpha, tol=args.tolerance)
    Γ = wall.surface_excess()
    w = wall.abs_deviation()
    γ, _, _ = wall.wall_tension()
    results.append((Awall, Γ, γ, w))

df = pd.DataFrame(results, columns=['Awall', 'Gamma', 'gamma', 'abs_dev'])

df['mN.m'] = df['gamma'] * args.conversion_factor # column for surface tension in physical units

if args.output:
    column_heads = [f'{col}({i+1})' for i, col in enumerate(df.columns)]
    header_row = '#  ' + '  '.join(column_heads)
    data_rows = df.to_string(index=False).split('\n')[1:]
    with open(args.output, 'w') as f:
        print('\n'.join([header_row] + data_rows), file=f)
    print('Data:', ', '.join(column_heads), 'written to', args.output)
else:
    print(df.set_index('Awall'))

    # Write in format suitable for import into xmgrace as NXY data

