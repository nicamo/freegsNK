#!/usr/bin/env python

import numpy as np
import freegs

#########################################
# Create the machine, which specifies coil locations
# and equilibrium, specifying the domain to solve over

tokamak = freegs.machine.MASTU()

#use initial psi_plasma
plasma_psi = np.loadtxt('test16/initialpsitest-nk.txt')
eq = freegs.Equilibrium(tokamak=tokamak,
                        Rmin=0.1, Rmax=2.0,    # Radial domain
                        Zmin=-2.2, Zmax=2.2,   # Height range
                        nx=129, ny=129, # Number of grid points
                        psi=plasma_psi)

#########################################
# Plasma profiles

profiles = freegs.jtor.ConstrainPaxisIp(6.1e3, # Plasma pressure on axis [Pascals]
                                        6.2e5, # Plasma current [Amps]
                                        0.5, # vacuum f = R*Bt
                                        alpha_m = 1.0,
                                        alpha_n = 2.0)


#########################################
# Load set of current values and get list of tokamak coils

current_vals = np.loadtxt('test16/valsfortimingtest-nk.txt')
coil_names = eq.tokamak.getCurrents().keys()





#########################################
# Solve forward problem, i.e. constrain=None
# on set of current values

for i in range(10):#len(current_vals)):
    # assign currents
    for j,key in enumerate(coil_names):
        eq.tokamak[key].current = current_vals[i,j]
    # solve
    freegs.solve(eq,          # The equilibrium to adjust
             profiles,        # The plasma profiles
             constrain=None,   # Plasma control constraints, this will use NK solver
             maxits=100,
             verbose=0,
             rtol=1e-6)    


input('Press enter to exit')
