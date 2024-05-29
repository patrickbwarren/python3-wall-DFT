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
import numpy as np
import pandas as pd
from wallDFT import wall_args, solve_args, df_header, df_to_agr

eparser = wallDFT.ExtendedArgumentParser(description='DFT wall property table calculator')
eparser.awall.default = '0,40,5'
eparser.awall.help='wall repulsion amplitudes, default ' + eparser.awall.default
eparser.add_argument('--ktbyrc2', default=12.928, type=float, help='kT/rc² = 12.928 mN.m')
eparser.add_argument('-o', '--output', help='output data to, eg, .dat')
args = eparser.parse_args()

Alo, Ahi, Astep = eval(args.Awall) # returns a tuple
Awalls = np.linspace(Alo, Ahi, round((Ahi-Alo)/Astep)+1, dtype=float)

wall = wallDFT.Wall(**wall_args(args))

if args.verbose:
    print(wall.about)

rhob = eval(args.rhob)
Abulk = eval(args.Abulk)

results = []

for Awall in Awalls:
    wall.continuum_wall(Awall*rhob) if args.continuum else wall.standard_wall(Awall)
    iters, conv = wall.solve(rhob, Abulk, **solve_args(args))
    Γ = wall.surface_excess()
    w = wall.abs_deviation()
    γ, _, _ = wall.wall_tension()
    result = (Awall, Γ, γ, w, conv, iters)
    if args.verbose > 1:
        print('%g\t%g\t%g\t%g\t%g\t%i' % result)
    results.append(result)

schema = {'Awall':float, 'Gamma':float, 'gamma':float,
          'abs_dev':float, 'conv':float, 'iters':int}
df = pd.DataFrame(results, columns=schema.keys()).astype(schema)

# column for wall tension in physical units
icol = df.columns.get_loc('gamma') + 1
df.insert(icol, 'mN.m', df['gamma'] * args.ktbyrc2)

if args.output:
    df.drop(['conv', 'iters'], axis=1, inplace=True)
    with open(args.output, 'w') as f:
        print(df_to_agr(df), file=f)
    print('Data:', ', '.join(df_header(df)), 'written to', args.output)
else:
    print(df.set_index('Awall'))

