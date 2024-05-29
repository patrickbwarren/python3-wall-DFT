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

import wall_dft
import numpy as np
import pandas as pd
from wall_dft import wall_args, solve_args, df_header, df_to_agr

eparser = wall_dft.ExtendedArgumentParser(description='DFT wall property table calculator')
eparser.rhob.default = '0,3.5,0.5'
eparser.rhob.help='bulk densities, default ' + eparser.rhob.default
eparser.add_argument('--rbase', default=3.0, type=float, help='baseline bulk density, default 3.0')
eparser.add_argument('--ktbyrc2', default=12.928, type=float, help='kT/rc² = 12.928 mN.m')
eparser.add_argument('-o', '--output', default=None, help='output data to, eg, .dat')
args = eparser.parse_args()

rholo, rhohi, rhostep = eval(args.rhob) # returns a tuple
rhobs = np.linspace(rholo, rhohi, round((rhohi-rholo)/rhostep)+1, dtype=float)

wall = wall_dft.Wall(**wall_args(args))

if args.verbose:
    print(wall.about)

Awall = eval(args.Awall)
Abulk = eval(args.Abulk)
rhob_base = args.rbase

wall.continuum_wall(Awall*rhob_base) if args.continuum else wall.standard_wall(Awall)

if args.verbose:
    print(wall.model)

result = (0.0, wall.curly_ell(), 0.0, 0.0, 0.0, 0)
if args.verbose > 1:
    print('%g\t%g\t%g\t%g\t%g\t%i' % result)
results = [result] # zero density result

for rhob in rhobs[1:]: # omit zero density
    iters, conv = wall.solve(rhob, Abulk, **solve_args(args))
    Γ = wall.surface_excess()
    w = wall.abs_deviation()
    γ, _, _ = wall.wall_tension()
    result = (rhob, Γ/rhob, γ, w, conv, iters)
    if args.verbose > 1:
        print('%g\t%g\t%g\t%g\t%g\t%i' % result)
    results.append(result)

schema = {'rhob':float, 'Gamma/rhob':float, 'gamma':float,
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
    print(df.set_index('rhob'))

