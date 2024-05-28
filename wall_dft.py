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
        self.rbulk = self.parser.add_argument('--rhob', default='3.0', help='bulk density, default 3.0')
        self.abulk = self.parser.add_argument('--Abulk', default='25', help='bulk repulsion amplitude, default 25')
        self.awall = self.parser.add_argument('--Awall', default='10', help='wall repulsion amplitude, default 10')
        self.parser.add_argument('-c', '--continuum', action='store_true', help='if set use a continuum half-space wall model')

    def add_argument(self, *args, **kwargs):
        return self.parser.add_argument(*args, **kwargs)

    def parse_args(self, *args, **kwargs):
        return self.parser.parse_args(*args, **kwargs)

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
        self.idx = np.round(self.z/self.dz).astype(int) # index with origin z = 0.0 → 0
        self.ρ = np.zeros_like(self.z)
        z = np.linspace(-1.0, 1.0, round(2.0/self.dz)+1, dtype=float)
        self.kernel = π/12.0*(1-z)**3*(1+3*z)
        self.kernel[z<0] = np.flip(self.kernel[z>0])
        self.about = 'Wall: zmax, dz, nz = %g, %g, %i' % (self.zmax, self.dz, len(self.z))

    def standard_wall(self, Awall):
        z = self.z
        self.uwall = 0.5*Awall*(1-z)**2
        self.uwall[z<0] = 0.5*Awall # continuity for z < 0 (though irrelevant)
        self.uwall[z>1] = 0.0
        self.model = f'standard wall: Awall = {Awall}'

    def continuum_wall(self, Awall_rhob):
        z = self.z
        self.uwall = π*Awall_rhob/60.0*(1-z)**4*(2+3*z)
        self.uwall[z<0] = π*Awall_rhob/30.0 # continuity for z < 0 (though irrelevant)
        self.uwall[z>1] = 0.0
        self.model = f'continuum wall: Awall*rhob = {Awall_rhob}'

    def truncate_to(self, a, v):
        b = a.copy()
        b[self.z < 0] = v
        return b

    def solve(self, rhob, Abulk, max_iters=300, alpha=0.1, tol=1e-10):
        z, dz, domain = self.z, self.dz, self.domain
        ukernel, uwall = (Abulk * self.kernel), self.uwall
        Δρ = self.truncate_to(rhob*(np.exp(-uwall) - 1), -rhob) # initial guess
        # Iterate to solve Δρ = ρb [exp(-Uext-Uself) - 1] where
        # Uself = ∫ dz' Δρ(z') U(z'−z).  Use convolution from
        # numpy to evaluate this integral.
        for i in range(max_iters):
            uself = dz * np.convolve(Δρ, ukernel, mode='same')
            Δρ_new = rhob * (np.exp(-uwall-uself) - 1)
            Δρ_new[z<0] = -rhob # always inside the hard wall barrier
            h0 = np.max(np.abs(Δρ))
            h1 = np.max(np.abs(Δρ_new))
            α = alpha * h0 / h1
            Δρ_new = (1-α)*Δρ + α*Δρ_new # mixing rule
            ΔΔρ = Δρ - Δρ_new
            int_abs_ΔΔρ = np.trapz(np.abs(ΔΔρ), dx=dz)
            if int_abs_ΔΔρ < tol: # early escape if converged
                break
            Δρ = Δρ_new
        self.Δρ = Δρ # The density profile (should vanish inside the wall)
        self.ρb = rhob # For use with surface tension calculation
        self.Abulk = Abulk # -- ditto --
        return i, int_abs_ΔΔρ # for monitoring

    def density_profile(self):
        return self.truncate_to(self.ρb + self.Δρ, 0)

    # The surface excess Γ/A = ∫ dz Δρ(z), where the integration
    # limits are 0 to ∞.  The absolute deviation is defined similarly
    # as ∫ dz |Δρ(z)| where the integration limits are 1 to ∞.

    def surface_excess(self):
        return np.trapz(self.Δρ[self.domain], dx=self.dz)

    def abs_deviation(self):
        return np.trapz(np.abs(self.Δρ[self.domain & ~(self.z<1)]), dx=self.dz)

    # Calculate the surface tension by first calculating the grand
    # potential per unit area with a certain domain height Lz.
    # The pressure is p = ρb + 1/2 ρb² ∫ dz U(z) and the grand
    # potential is -Ω/A = ∫ dz ρ(z) + 1/2 ∫ dz dz' ρ(z) ρ(z')
    # U(z-z').  The z-integration limits are 0 to ∞ (but in any
    # case the density ρ(z) vanishes for z < 0).  The function
    # U(z) is the integrated DPD potential (ukernel).  With these
    # quantities the surface tension γ = p*Lz + Ω/A.
    # (Alternatively the pressure is the negative of the grand
    # potential density, p = -ω, in the homogeneous system.)  In
    # the calculation below the contribution Lz*ρb is omitted from
    # both the terms since it cancels in the expression for the
    # surface tension.
        
    # For a matched continuum wall potential ρ(z) = ρb for z > 0, so
    # that the surface excess vanishes.  However γ = π A ρb²/240 ; for
    # the standard water model (A = 25, ρb = 3), this is γ = 15π/16 ≈
    # 2.94524.

    def wall_tension(self):
        z, dz, domain = self.z, self.dz, self.domain
        ukernel, Δρ = (self.Abulk * self.kernel), self.Δρ
        ρ = self.truncate_to(self.ρb + Δρ, 0)
        ρρU = ρ * dz * np.convolve(ρ, ukernel, mode='same')
        ΩexbyA = - np.trapz(Δρ[domain], dx=dz) - 0.5 * np.trapz(ρρU[domain], dx=dz) # omitting Lz*ρb
        ωbex = - 0.5 * self.ρb**2 * np.trapz(ukernel, dx=dz) # same as np.conv, omitting Lz*ρb
        ωb = ωbex - self.ρb # restoring the bulk piece
        # Lz = np.trapz(np.ones_like(z[domain]), dx=dz) 
        Lz = z[domain][-1] - z[domain][0] # equivalent to above
        γ = ΩexbyA - Lz*ωbex
        return γ, ωb, Lz
