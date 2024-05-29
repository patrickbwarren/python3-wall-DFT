## Density functional theory for walls in dissipative particle dynamics

### Summary

This code computes the density profile, surface excess, and wall
tension, for a fluid of pure dissipative particle dynamics (DPD)
beads against a wall, using a mean-field density functional theory (DFT).

The codes are

* `wall_dft_single.py` : solve a single state point;
* `wall_dft_awall_scan.py` : solve for an array of wall repulsion amplitudes;
* `wall_dft_rhob_scan.py` : solve for an array of bulk densities;
* `wall_dft_zero.py` : solve for zero surface excess or wall tension;
* `wall_dft_minim.py` : solve for minimum perturbation to bulk;
* `wallDFT.py` : underlying python module implementing functionality.

All codes can be run without command line parameters, using the
built-in defaults.  

### What's being solved here?

Consider a fluid of particles at a non-uniform density ρ(**r**)
interacting with a soft, short-range potential *U*(*r*), in the
presence of an external potential *U*<sub>ext</sub>(**r**). A
mean-field density functional free energy for this system is

* ∫ ρ(**r**) [ln ρ(**r**) − 1] + ∫ ρ(**r**) *U*<sub>ext</sub>(**r**) +
  1/2 ∫ ρ(**r**) ρ(**r**') U(|**r**−**r**'|) .
  
If we consider that *U*<sub>ext</sub> = *U*<sub>wall</sub> represents
a wall in the plane normal to the *z*-direction, then this reduces to

* ∫ ρ(*z*) [ln ρ(*z*) − 1] + ∫ ρ(*z*) *U*<sub>wall</sub>(*z*) + 1/2 ∫
  ρ(*z*) ρ(*z*') *U*(*z*−*z*') ,

where *U*(*z*) is a partial integral of *U*(**r**) corresponding to
the interaction between two sheets of particles at unit density,
separated by a distance *z*.

The corresponding grand potential (per unit area) is given by the
above expression − μ ∫ ρ(*z*), where μ is the chemical potential.  By
eliminating μ in favour of the bulk density ρ<sub>b</sub>, and
minimising the grand potential, one arrives at the following condition
for the density profile in the wall potential,

* ρ(*z*) = ρ<sub>b</sub> exp[ − *U*<sub>wall</sub>(*z*) − ∫ d*z*'
  [ρ(*z*') − ρ<sub>b</sub>] U(*z*−*z*') ] .

This has to be solved self-consistently, and this is what the code in
this repository does. As input, one needs to specify the bulk density
and the two potential functions *U*(*z*) and *U*<sub>wall</sub>(*z*).

The method is to start with the initial guess ρ(*z*) = ρ<sub>b</sub>
exp[ − *U*<sub>w</sub>(*z*) ], and iterate the above, but mixing in
only a fraction of the new solution at each iteration step to assure
convergence (Picard method).  The integral can be evaluated as a
convolution, using a standard numerical library routine.

#### Thermodynamics

Given a solution ρ(*z*), one can compute the surface excess Γ and the
wall tension γ.  The former is just the integral of Δρ(*z*) where

* Δρ(*z*) = ρ(*z*) − ρ<sub>b</sub> , 

and the latter is the excess grand potential per unit area. The bulk
grand potential per unit volume needed here is just Ω / V = −*p*, where *p*
is the mean-field pressure,

* *p* = ρ<sub>b</sub> + 1/2 ρ<sub>b</sub><sup>2</sup> ∫
d<sup>3</sup>**r** U(**r**)

(cf first expression above).  To minimise the perturbation caused by
the wall, one can also define the perturbation in the bulk as the
integral of |Δρ(*z*)| outside of the wall potential.

####  Dissipative particle dynamics

In DPD the interaction potential *U*(*r*) = A/2 (1−r)<sup>2</sup> for
*r* < 1, and *U*(*r*) = 0 for *r* > 1.  We set the range to unity to
set the length scale, and work in units where the thermal energy scale
*k*<sub>B</sub>*T* = 1.  This gives the following partial integral to
be used in the above

* *U*(*z*) = π A / 12 (1 − *z*)<sup>3</sup> (1 + 3*z*) ,

 for 0 ≤ *z* ≤ 1, together with *U*(*z*) = 0 for *z* > 1, and
 *U*(−*z*) = *U*(*z*) for *z* < 0.

#### Wall models

We consider two different wall models, a 'standrd' one where

* 


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
