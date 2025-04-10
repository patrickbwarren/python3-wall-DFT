## Density functional theory for walls in dissipative particle dynamics

### Summary

This code computes the density profile, surface excess, and wall
tension, for a fluid of pure dissipative particle dynamics (DPD)
beads against a wall, using a mean-field density functional theory (DFT).

The codes are:

* `wall_dft_single.py` : solve a single state point;
* `wall_dft_awall_scan.py` : solve for an array of wall repulsion amplitudes;
* `wall_dft_rhob_scan.py` : solve for an array of bulk densities;
* `wall_dft_zero.py` : solve for zero surface excess or wall tension;
* `wall_dft_minim.py` : solve for minimum perturbation to bulk;
* `wallDFT.py` : underlying python module implementing functionality.

All the python scripts can be run without command line parameters,
using the built-in defaults. They were developed to support a
forthcoming publication in the area. 

Any other scripts in this repository are experimental; you are welcome
to investigate them and use them, but they are not particularly well
documented !  See:

* `lle_interface.py` : solve liquid-liquid phase equilibria.

### What's being solved here?

Consider a fluid of particles at a non-uniform density ρ(**r**)
interacting with a pair potential *U*(*r*), in the presence of an
external potential *U*<sub>ext</sub>(**r**). A mean-field density
functional free energy for this system is

* ∫ ρ(**r**) [ln ρ(**r**) − 1] + ∫ ρ(**r**) *U*<sub>ext</sub>(**r**) +
  ½ ∫ ρ(**r**) ρ(**r**') U(|**r**−**r**'|) .
  
If we consider the case where *U*<sub>ext</sub>(**r**) =
*U*<sub>wall</sub>(*z*) represents a wall in the plane normal to the
*z*-direction, then this reduces to

* ∫ ρ(*z*) [ln ρ(*z*) − 1] + ∫ ρ(*z*) *U*<sub>wall</sub>(*z*) + ½ ∫
  ρ(*z*) ρ(*z*') *U*(*z*−*z*') ,

where 

* *U*(*z*) = 2π ∫<sub>|*z*|</sub> *r* d*r* *U*(*r*)

is the partial integral (from *r* = |*z*| to ∞) of *U*(**r**)
corresponding to the interaction between two parallel sheets of
particles at unit density, separated by a distance *z*.

The corresponding grand potential (per unit area) is given by
subtracting μ ∫ ρ(*z*) from the above expression, where μ is the
chemical potential.  By eliminating μ in favour of the bulk density
ρ<sub>b</sub>, and minimising the grand potential, one arrives at the
following condition for the density profile in the wall potential,

* ρ(*z*) = ρ<sub>b</sub> exp[ − *U*<sub>wall</sub>(*z*) − ∫ d*z*'
  Δρ(*z*') U(*z*−*z*') ] ,
  
where Δρ(*z*) = ρ(*z*) − ρ<sub>b</sub>. This has to be solved
self-consistently, and this is what the code in this repository
does. As input, one needs to specify the bulk density and the two
potential functions *U*(*z*) and *U*<sub>wall</sub>(*z*).  The method
is to start with the initial guess ρ(*z*) = ρ<sub>b</sub> exp[ −
*U*<sub>wall</sub>(*z*) ], and iterate the above, mixing in a fraction of
the new solution at each iteration step to assure convergence (Picard
method).  The integral in the exponential here can be evaluated as a
convolution, using a standard numerical library routine.

#### Wall tension and surface excess

Given a solution ρ(*z*), one can compute the wall tension γ and the
surface excess Γ. The former is just the excess grand potential per
unit area, and I define the latter as ∫ d*z* Δρ(*z*) outside the hard
wall (see below) (*z* ≥ 0). The grand potential per unit area Ω / *A*
= ∫ d*z* ρ(*z*) ω<sub>N</sub>(*z*) where ω<sub>N</sub>(z) = − 1 − ½ ∫
d*z*' ρ(*z*') *U*(*z*−*z*').  The bulk grand potential per unit volume
needed to calculate γ is ω<sub>b</sub> = − *p* where *p* =
ρ<sub>b</sub> + ½ ρ<sub>b</sub><sup>2</sup> ∫ d*z* *U*(*z*) is the pressure.

It follows from classical thermodynamics that dγ = − Γ dμ (Gibbs
isotherm). It is also true that d*p* = ρ dμ (Gibbs-Duhem relation).
Hence the dγ/d*p* = − Γ / ρ , and this can be verified numerically.
Note that Γ / ρ, which can be positive or negative, can be interpreted
as an adsorption length.

To quantify the perturbation in the bulk caused by the wall, the codes
also report the integral of |Δρ(*z*)| outside of the wall potential
(*z* > 1 in the wall models defined below).

####  Dissipative particle dynamics

In DPD the pair potential is

* *U*(*r*) = A (1−r)<sup>2</sup> / 2 

for *r* < 1, and *U*(*r*) = 0 for *r* > 1. The length scale is set by
the range *r*<sub>c</sub> = 1, and the energy scale is set by the
choice *k*<sub>B</sub>*T* = 1. The DPD model is then characterised by
the bulk density ρ<sub>b</sub> and repulsion amplitude *A*.  For
example, the standard DPD water model has ρ<sub>b</sub> = 3 and *A* =
25.

This choice of pair potential gives the following partial integral to
be used in the wall DFT calculations,

* *U*(*z*) = π A (1 − *z*)<sup>3</sup> (1 + 3*z*) / 12

 for 0 ≤ *z* ≤ 1, together with *U*(*z*) = 0 for *z* > 1, and
 *U*(*z*) = *U*(|*z*|) for *z* < 0.

#### Wall models

I consider two different wall models, a 'standard' one where

* *U*<sub>wall</sub>(*z*) = *A*<sub>wall</sub> (1 − *z*)<sup>2</sup> /
  2

for 0 ≤ *z* ≤ 1, together with *U*<sub>wall</sub>(*z*) = 0 for *z* >
 1, and *U*<sub>wall</sub>(*z*) = ∞ for *z* < 0 (ie a hard repulsive
 barrier at *z* = 0); and a minimally perturbative 'continuum' wall
 model with

* *U*<sub>wall</sub>(*z*) = π *A*<sub>wall</sub> ρ<sub>wall</sub> (1 −
  z)<sup>4</sup> (2 + 3*z*) / 60

for 0 ≤ *z* ≤ 1, together with similarly *U*<sub>wall</sub>(*z*) = 0
for *z* > 1, and *U*<sub>wall</sub>(*z*) = ∞ for *z* < 0.  Note that
for *z* > 0 the corresponding wall force − ∂*U*<sub>wall</sub> / ∂*z*
= ρ<sub>b</sub> *U*(*z*) for *U*(*z*) given above.

One can show that in the latter case, for *A*<sub>wall</sub>
ρ<sub>wall</sub> = *A* ρ<sub>b</sub>, the mean-field DFT density
profile is flat up to the wall, with ρ(*z*) = ρ<sub>b</sub> for 
*z* > 0.  Under these conditions, the wall is minimally perturbative
in the sense that Δρ(*z*) = 0 outside the wall.  Trivially, the
surface excess Γ = 0.  One can show that the wall tension calculated
from the excess mean-field grand potential per unit area in this case
is γ = π *A* ρ<sub>b</sub><sup>2</sup> / 240.

The continuum wall model derives its name because it also corresponds
to a DPD particle interacting with a continuum of wall particles at a
density ρ<sub>wall</sub> in the *z* < 0 half-space.  This is a model
first introduced by Goicochea and Alarcón in [J. Chem. Phys. **134**,
014703 (2011)](https://doi.org/10.1063/1.3517869).

For early work on minimally-pertubative wall models in a more generic
setting see Henderson in [Mol. Phys. **74**, 1125
(1991)](https://doi.org/10.1080/00268979100102851).  I am grateful to
Bob Evans for drawing my attention to this.

### Liquid-liquid phase equilibria

The code `lle_interface.py` calculates the density profiles and
interfacial properties (surface excesses, surface tension) for
coexisting phases in a binary DPD mixture.  The methods are similar to
those above, except there is no wall potential and there are two
density fields.  The trickiest part is to solve for the phase
equilibrium.  This is done assuming NPT, where the pressure is set by
specifying a baseline (default ρ = 3, *A* = 25).  For fixed pressure
there are two coexisting phase compositions *x*<sub>1</sub> and
*x*<sub>1</sub>.  These are obtained by solving numerically for
equality of chemical potentials in the coexisting phases.  The initial
part of the code computes these compositions, and the associated
chemical potentials of the two components (the same in both phases).
These are then used in the density profile calculation.

### Use of direct correlation function

The code `dpd_cr.py` investigtes using the direct corelation function
*c*(*r*) in the DFT rather than *U*(*r*), using `pyHNC.py`
from
[python3-HNC-solver](https://github.com/patrickbwarren/python3-HNC-solver)
to solve for this.  Preliminary work at present.

### Copying

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see
<http://www.gnu.org/licenses/>.

### Copyright

This program is copyright &copy; 2024 Patrick B Warren (STFC).  

### Contact

Send email to patrick.warren{at}stfc.ac.uk.
