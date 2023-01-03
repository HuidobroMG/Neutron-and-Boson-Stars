# -*- coding: utf-8 -*-
"""
@author: HuidobroMG

Description:
    
    This codes solves the TOV system of differential equations, which 
    describes the properties of a static spherically symmetric neutron star.
    It requires an EOS in a file.dat format, in which the first column is 
    the pressure, the second is the energy density, the third is the baryon
    density and the last one (which is not necessary in this case) is the
    derivative of the energy density w.r.t the pressure. It can run a single
    star for a concrete value of the pressure in the centre of the star
    (NumStars = 1) or it can run the whole Mass-Radius curve (NumStars = 0).
    A crust may be added at some value of p_PT, by default the crust is
    described by the BCPM EOS.
        
"""

#-----------------------------------------------------------------------------

# Packages
import numpy as np
import scipy.integrate as si # To solve the TOV system
import scipy.interpolate as scinter # For Nuclear Physics EoS
import matplotlib.pyplot as plt # Plots
import Units

#-----------------------------------------------------------------------------

# Read the data
name = 'BCPM'
data = np.loadtxt('EoS_Data/EoS_'+name+'.dat').T

# Interpolate the EOS
f = scinter.interp1d(data[0], data[1])
g = scinter.interp1d(data[0], data[2])
df = scinter.interp1d(data[0], data[3])

pmax = 0.9*max(data[0])
pmin = 1.1*min(data[0])
pmx = pmax*Units.MeV_fm3_2_Msun_km3
pmn = pmin*Units.MeV_fm3_2_Msun_km3

# Add a crust (not necessary)
p_PT = 0 # MeV/fm3
quadratic_interp = False

if p_PT != 0:
    dataB = np.loadtxt('EoS_Data/EoS_BCPM.dat').T
    fB = scinter.interp1d(dataB[0], dataB[1])
    gB = scinter.interp1d(dataB[0], dataB[2])
    dfB = scinter.interp1d(dataB[0], dataB[3])
    pmin = 1.1*min(dataB[0])
    if quadratic_interp == True:
        pmax = min(max(data[0]), max(dataB[0]))
        pmx = pmax*Units.MeV_fm3_2_Msun_km3
        p = np.exp(np.arange(np.log(min(dataB[0])), np.log(pmax)+0.1, 0.1))
        p[0] = min(dataB[0])
        p[-1] = pmax
        pmax = min(0.9*max(data[0]), 0.9*max(dataB[0]))
        pmx = pmax*Units.MeV_fm3_2_Msun_km3
        n = np.zeros(len(p))
        n[0] = gB(p[0])
        for i in range(len(p)-1):
            delta_p = p[i+1] - p[i]
            alpha = (p[i]/p_PT)**2/(1 + (p[i]/p_PT)**2)
            dalpha = 2*p[i]/p_PT**2/(1 + (p[i]/p_PT)**2)**2
            rho = (1-alpha)*fB(p[i]) + alpha*f(p[i])
            drho_dp = dalpha*(f(p[i]) - fB(p[i])) + (1-alpha)*dfB(p[i]) + alpha*df(p[i])
            n[i+1] = n[i] + delta_p*drho_dp*n[i]/(p[i]+rho)
        n_func = scinter.interp1d(p, n)
    if p_PT >= 1e5 and quadratic_interp == False:
        pmax = 0.9*max(dataB[0])
        pmx = pmax*Units.MeV_fm3_2_Msun_km3

# Amount of stars
NumStars = 0

#-----------------------------------------------------------------------------

# EOS function
def EoS(p):
    p /= Units.MeV_fm3_2_Msun_km3
    
    rho = f(p)
    n = g(p)
    if quadratic_interp == True:
        alpha = (p/p_PT)**2/(1 + (p/p_PT)**2)
        rho = (1-alpha)*fB(p) + alpha*f(p)
        n = n_func(p)
    else:
        if p <= p_PT:
            rho = fB(p)
            n = gB(p)
    return np.array([rho*Units.MeV_fm3_2_Msun_km3, n/Units.fm_2_km**3])

#-----------------------------------------------------------------------------

# TOV Equations
def TOV(r, var):
    A, B, p, N = var
    
    if p <= pmn:
        p = 0
        rho = 0
        n = 0
    else:
        rho, n = EoS(p)
    
    if r == 0.0:
        du1 = 0.0
    else:
        du1 = 1.0/r

    dA = A*(Units.kappa*r*B*p + (B-1.0)*du1)
        
    dB = B*(Units.kappa*r*B*rho - (B-1.0)*du1)
    
    dp = -(rho + p)*dA/(2*A)
        
    dN = 4.0*np.pi*r**2*n*np.sqrt(B)/Units.Nsun
    return np.array([dA, dB, dp, dN])

#-----------------------------------------------------------------------------

# Mass calculation function from the metric grr function
def Mass_Value(R, B):
    M = R/(2.0*Units.K)*(1.0-1.0/B)
    return M

#------------------------------------------------------------------------------

# Parameters
R0 = 0.0
Rf = 30.0
step = 1e-3

r = np.arange(R0, Rf, step)

A0 = 1.0 # Irrelevant
B0 = 1.0
N0 = 0.0

#------------------------------------------------------------------------------

#Run one star

# Initial conditions
p0 = 1.5*(pmax+pmin)/2*Units.MeV_fm3_2_Msun_km3
rho0, n0 = EoS(p0)

vinic = [A0, B0, p0, N0]


sol = si.solve_ivp(TOV, (R0, Rf), vinic, method = 'Radau', t_eval = r,
                   rtol = 1e-10)

A, B, p, N = sol.y
rad = sol.t

if np.any(p <= 0):
    idx = np.where(p < 1e-15)[0][0]
    if p[idx] <= 0:
        idx -= 1
else:
    idx = np.where(p == min(p))[0][0]

Radius = rad[idx]
Mass = Mass_Value(Radius, B[idx])
Baryon = N[idx]

rho = []
for i in range(len(p[:idx])):
    rho.append(EoS(p[i])[0])

rho = np.array(rho)
    
# Calculate the conformal asymmetry
Delta = 1/3 - p[:idx]/rho

#------------------------------------------------------------------------------

# Run the whole MR curve
def Star(p0):
    
    vinic = [A0, B0, p0, N0]
    
    sol = si.solve_ivp(TOV, (R0, Rf), vinic, method = 'Radau', t_eval = r,
                       rtol = 1e-10)
    
    A, B, p, N = sol.y
    rad = sol.t
    
    if np.any(p <= 0):
        idx = np.where(p < 1e-15)[0][0]
        if p[idx] <= 0:
            idx -= 1
    else:
        idx = np.where(p == min(p))[0][0]
    
    Radius = rad[idx]
    Mass = Mass_Value(Radius, B[idx])
    Baryon = N[idx]
    
    return Radius, Mass, Baryon

if NumStars == 0:
    p0 = np.exp(np.linspace(np.log(pmx/500), np.log(pmx), 25))
    
    Mass = []
    Radius = []
    Baryon = []
    
    i = 0
    Nstars = len(p0)
    for p in p0:
        i += 1
        print(str(i)+'/'+str(Nstars))
        R, Ms, Nb, = Star(p)
        Radius.append(R)
        Mass.append(Ms)
        Baryon.append(Nb)

np.savetxt('MR_'+name+'.dat', np.array([Radius, Mass, Baryon]).T)
