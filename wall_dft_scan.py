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

Alo, Ahi, Astep = eval(args.Awall)
Awalls = np.linspace(Alo, Ahi, round((Ahi-Alo)/Astep)+1, dtype=float)

wall = wall_dft.Wall(dz=args.dz, zmax=args.zmax)
print(wall.about)

wall.rhob = eval(args.rhob)
wall.Abulk = eval(args.Abulk)
wall.model = 'half space' if args.half_space else 'vanilla'

df = pd.DataFrame(columns=['Awall', 'surface_excess', 'gamma', 'wobble'], dtype='float') # initially empty

for i, wall.Awall in enumerate(Awalls):
    wall.solve(max_iters=max_iters, alpha=args.alpha, tol=args.tolerance)
    df.loc[i] = (wall.Awall, wall.surface_excess, wall.gamma, wall.wobble)

df['mN.m'] = df['gamma'] * args.conversion_factor # column for surface tension in physical units

if args.output:
    df.to_csv(args.output, sep='\t', header=False, index=False, float_format='%g')
    print(', '.join([f'{s}({i+1})' for i, s in enumerate(df.columns)]), 'written to', args.output)
else:
    print(df.set_index('Awall'))
