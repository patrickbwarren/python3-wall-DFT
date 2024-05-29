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

### Copying

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

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
