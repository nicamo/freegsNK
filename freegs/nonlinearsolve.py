"""
Routines for solving the nonlinear part of the Grad-Shafranov equation

Copyright 2016-2019 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

This file is part of FreeGS.

FreeGS is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGS.  If not, see <http://www.gnu.org/licenses/>.
"""

# modified by nicamo
# call to solve is changed so that
# if solve is used for forward problem, ie constrain=None
# then newtonk.NKsolve is called instead
# if solve is used for inverse problem
# picard (i.e. original solve function) is used

from numpy import amin, amax, array

from .picard import Psolve
from .newtonk import NKsolve

def solve(
    eq,
    profiles,
    constrain=None,
    rtol=1e-3,
    atol=1e-10,
    blend=0.0,
    show=False,
    axis=None,
    pause=0.0001,
    psi_bndry=None,
    maxits=50,
    verbose=0,
    convergenceInfo=False
):

   
    if constrain is not None:
        # use Picard solver
        Psolve(  eq,
                profiles,
                constrain,
                rtol,
                atol,
                blend,
                show,
                axis,
                pause,
                psi_bndry,
                maxits,
                verbose,
                convergenceInfo
            )
        
    else:
        # use Newton Krylov solver
        NKsolve(eq, 
          profiles,
          rtol=rtol,
          atol=atol,
          show=show,
          axis=axis,
          pause=pause,
          n_k=8,
          new_t=.2, 
          grad_eps=1,
          clip=3,
          maxits=maxits,
          verbose=verbose,
          convergenceInfo=convergenceInfo
          )


