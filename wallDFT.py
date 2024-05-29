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

import argparse
import numpy as np
from numpy import pi as π

# parser.add_argument returns a handle to the argument object which
# can be attached to parser itself as an attribute.  This can be
# accessed to replace the default value and help string.

class ExtendedArgumentParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        self.parser = argparse.ArgumentParser(*args, **kwargs)
        self.parser.add_argument('--max-iters', default='10^3', help='max number of iterations, default 10^3')
        self.parser.add_argument('--tolerance', default='1e-10', type=float, help='convergence tol, default 1e-10')
        self.parser.add_argument('--alpha', default=0.1, type=float, help='mixing fraction, default 0.1')
        self.parser.add_argument('--zmax', default=11.0, type=float, help='maximum distance in z, default 10.0')
        self.parser.add_argument('--dz', default=1e-3, type=float, help='spacing in z, default 1e-3')
        self.rhob = self.parser.add_argument('--rhob', default='3.0', help='bulk density, default 3.0')
        self.rhow = self.parser.add_argument('--rhow', default='3.0', help='wall density, default 3.0')
        self.abulk = self.parser.add_argument('--Abulk', default='25', help='bulk repulsion amplitude, default 25')
        self.awall = self.parser.add_argument('--Awall', default='10', help='wall repulsion amplitude, default 10')
        self.add_argument('--ktbyrc2', default=12.928, type=float, help='kT/rc² = 12.928 mN.m')
        self.parser.add_argument('-c', '--continuum', action='store_true', help='if set use a continuum half-space wall')
        self.parser.add_argument('-v', '--verbose', action='count', default=0, help='increasing verbosity')

    def add_argument(self, *args, **kwargs):
        return self.parser.add_argument(*args, **kwargs)

    def parse_args(self, *args, **kwargs):
        return self.parser.parse_args(*args, **kwargs)

def wall_args(args):
    '''Return a dict of generic args that can be used as **wall_args(args)'''
    return {'dz': args.dz, 'zmax': args.zmax}

def solve_args(args):
    '''Return a dict of generic args that can be used as **solve_args(args)'''
    max_iters = eval(args.max_iters.replace('^', '**'))
    return {'max_iters': max_iters, 'alpha': args.alpha, 'tol': args.tolerance}

def truncate_to_zero(a, z):
    '''Truncate an array to zero for z < 0'''
    b = a.copy()
    b[z < 0] = 0.0
    return b

class Wall:

    # Define the kernel U(z) on the domain -1 < z < 1.  The function is
    # π/12 (1−z)³(1+3z) for 0 < z < 1 and U(-z) = U(z) for -1 < z < 0.
    # The domain here includes the ends, ie z ∈ [-1, 1].  This means that
    # np.trapz is equivalent to np.conv since the endpoints are zero.  We
    # omit the 'A' factor, and restore it when solving for a density
    # profile.

    def __init__(self, dz=1e-3, zmax=11.0):
        self.dz = dz
        self.zmax = zmax
        self.z = np.linspace(-1.0, zmax, round((1.0+zmax)/dz)+1, dtype=float) # z ∈ [-1, zmax]
        self.not_inside_wall = ~(self.z < 0) # binary array
        self.not_above_zmax_minus_one = ~(self.z > self.zmax - 1.0) # ditto
        self.domain = self.not_inside_wall & self.not_above_zmax_minus_one # ditto
        self.idx = np.round(self.z/self.dz).astype(int) # index with origin z = 0.0 --> 0
        z = np.linspace(-1.0, 1.0, round(2.0/self.dz)+1, dtype=float)
        self.kernel = π/12.0*(1-z)**3*(1+3*z)
        self.kernel[z<0] = np.flip(self.kernel[z>0])
        self.about = 'Wall: zmax, dz, nz = %g, %g, %i' % (self.zmax, self.dz, len(self.z))

    def standard_wall(self, Awall):
        z = self.z
        self.uwall = 0.5*Awall*(1-z)**2
        self.uwall[z<0] = np.inf # hard repulsive barrier
        self.uwall[z>1] = 0.0
        self.expneguwall = truncate_to_zero(np.exp(-self.uwall), z)
        self.model = f'Standard wall: Awall = {Awall}'

    def continuum_wall(self, Awall_rhow):
        z = self.z
        self.uwall = π*Awall_rhow/60.0*(1-z)**4*(2+3*z)
        self.uwall[z<0] = np.inf # hard repulsive barrier
        self.uwall[z>1] = 0.0
        self.expneguwall = truncate_to_zero(np.exp(-self.uwall), z)
        self.model = f'Continuum wall: Awall*rhow = {Awall_rhow}'

    def curly_ell(self): # available after the wall potential is set
        return np.trapz((self.expneguwall[self.domain] - 1.0), dx=self.dz)

    def solve(self, rhob, Abulk, max_iters=300, alpha=0.1, tol=1e-10, eps=1e-10):
        z, dz, domain = self.z, self.dz, self.domain
        ukernel, uwall = (Abulk * self.kernel), self.uwall
        Δρ = rhob * (self.expneguwall - 1) # initial guess
        # Iterate to solve Δρ = ρb [exp(-Uext-Uself) - 1] where
        # Uself = ∫ dz' Δρ(z') U(z'−z).  Use convolution from
        # numpy to evaluate this integral.
        for i in range(max_iters):
            uself = dz * np.convolve(Δρ, ukernel, mode='same')
            Δρ_new = rhob * (self.expneguwall*np.exp(-uself) - 1.0)
            h0, h1 = [np.max(np.abs(a)) for a in [Δρ, Δρ_new]]
            α = alpha * h0 / (h1 + eps)
            Δρ_new = (1-α)*Δρ + α*Δρ_new # mixing rule
            ΔΔρ = Δρ - Δρ_new
            int_abs_ΔΔρ = np.trapz(np.abs(ΔΔρ), dx=dz)
            if int_abs_ΔΔρ < tol: # early escape if converged
                break
            Δρ = Δρ_new
        self.Δρ = Δρ # The density profile (should vanish inside the wall)
        self.ρb = rhob # For use with wall tension calculation
        self.Abulk = Abulk # -- ditto --
        return i, int_abs_ΔΔρ # for monitoring

    def density_profile(self):
        return truncate_to_zero(self.ρb + self.Δρ, self.z)

    # The surface excess Γ/A = ∫ dz Δρ(z), where the integration
    # limits are 0 to ∞.  The absolute deviation is defined similarly
    # as ∫ dz |Δρ(z)| where the integration limits are 1 to ∞.

    def surface_excess(self):
        return np.trapz(self.Δρ[self.domain], dx=self.dz)

    def abs_deviation(self):
        return np.trapz(np.abs(self.Δρ[self.domain & ~(self.z<1)]), dx=self.dz)

    # Calculate the wall tension by first calculating the grand
    # potential per unit area with a certain domain height Lz.
    # The pressure is p = ρb + 1/2 ρb² ∫ dz U(z) and the grand
    # potential is -Ω/A = ∫ dz ρ(z) + 1/2 ∫ dz dz' ρ(z) ρ(z')
    # U(z-z').  The z-integration limits are 0 to ∞ (but in any
    # case the density ρ(z) vanishes for z < 0).  The function
    # U(z) is the integrated DPD potential (ukernel).  With these
    # quantities the wall tension γ = p*Lz + Ω/A.

    # (Alternatively the pressure is the negative of the grand
    # potential density, p = -ω, in the homogeneous system.)  In
    # the calculation below the contribution Lz*ρb is omitted from
    # both the terms since it cancels in the expression for the
    # wall tension.

    def wall_tension(self):
        z, dz, domain = self.z, self.dz, self.domain
        ukernel, Δρ = (self.Abulk * self.kernel), self.Δρ
        ρ = truncate_to_zero(self.ρb + Δρ, z)
        ρρU = ρ * dz * np.convolve(ρ, ukernel, mode='same')
        ΩexbyA = - np.trapz(Δρ[domain], dx=dz) - 0.5 * np.trapz(ρρU[domain], dx=dz) # omitting Lz*ρb
        ωbex = - 0.5 * self.ρb**2 * np.trapz(ukernel, dx=dz) # same as np.conv, omitting Lz*ρb
        ωb = ωbex - self.ρb # restoring the bulk piece
        # Lz = np.trapz(np.ones_like(z[domain]), dx=dz) 
        Lz = z[domain][-1] - z[domain][0] # equivalent to above
        γ = ΩexbyA - Lz*ωbex
        return γ, ωb, Lz

    def solve_and_print(self, Awall, rhow, Abulk, rhob, args):
        self.continuum_wall(Awall*rhow) if args.continuum else self.standard_wall(Awall)
        iters, conv = self.solve(rhob, Abulk, **solve_args(args))
        Γ = self.surface_excess()
        w = self.abs_deviation()
        γ, ωb, Lz = self.wall_tension()
        p_mf = rhob + π/30 * Abulk * rhob**2
        print(self.model)
        print('Converged after %i iterations, ∫dz|ΔΔρ| = %g' % (iters, conv))
        print('Awall, ρw, Abulk, ρb = %g, %g, %g, %g' % (Awall, rhow, Abulk, rhob))
        print('Bulk grand potential ωb = %g' % ωb)
        print('Bulk mean field pressure, p = %g' % p_mf)
        print('Domain size Lz = %g' % Lz)
        print('Abs deviation in bulk = %g' % w)
        print('Surface excess per unit area Γ/A = %g' % Γ)
        print('Wall tension γ = %g = %g mN.m' % (γ, γ * args.ktbyrc2))

# Make the data output suitable for plotting in xmgrace if captured by redirection
# stackoverflow.com/questions/30833409/python-deleting-the-first-2-lines-of-a-string

def df_header(df):
    '''Generate a header of column names as a list'''
    return [f'{col}({i+1})' for i, col in enumerate(df.columns)]

def df_to_agr(df):
    '''Convert a pandas DataFrame to a string for an xmgrace data set'''
    header_row = '#  ' + '  '.join(df_header(df))
    data_rows = df.to_string(index=False, float_format='%g').split('\n')[1:]
    return '\n'.join([header_row] + data_rows)
