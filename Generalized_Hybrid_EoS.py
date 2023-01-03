# -*- coding: utf-8 -*-
"""
@author: HuidobroMG

Description:
    
    This code constructs the Generalized and Hybrid EOS which results from
    the combination of the standard and BPS Skyrme model.
        
"""

#-----------------------------------------------------------------------------

# Packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scinter

#------------------------------------------------------------------------------

# Possible values of p_PT: 25, 40, 50
p_core = 50
# Possible values of p_*: 0.5, 1, 2
p_crust = 0.5

# Array of pressure
p = np.exp(np.linspace(np.log(5.958e-15), np.log(1500), 200)) # MeV/fm3

#------------------------------------------------------------------------------

# Standard Skyrme crystal EoS
alpha = 0.474
beta = 0.0515
E0 = 923.3/(2.0*alpha + beta) # MeV
L0 = 1.0/0.16**(1.0/3.0) # fm
alfa1 = E0*alpha*L0/3.0
alfa2 = E0*alpha/(3.0*L0)

y = (alfa2 + np.sqrt(alfa2**2+4.0*alfa1*p))/(2.0*alfa1)
l = 1.0/np.sqrt(y)
dy = 1.0/np.sqrt(alfa2**2+4.0*alfa1*p)
dl = -0.5*dy/y**1.5

rho_Sk = E0*(alpha*(l/L0+L0/l)+beta)/l**3
n_Sk = y**1.5
drhodp_Sk = -2.0*E0*alpha*(1.0/(L0*l**3) + 2.0*L0/l**5)*dl

#------------------------------------------------------------------------------

# Effective BPS Skyrme model EoS
yc = (alfa2 + np.sqrt(alfa2**2+4.0*alfa1*p_core))/(2.0*alfa1)
lc = 1.0/np.sqrt(yc)
rho_Skc = E0*(alpha*(lc/L0+L0/lc)+beta)/lc**3

rho_BPS = p + rho_Skc
n_BPS = 0 # We do not care about the effective BPS baryon density
drhodp_BPS = np.ones(len(p)) # The derivative is trivial

#------------------------------------------------------------------------------

# Generalized model
beta_core = 0.9
Interp = (p/p_core)**beta_core/(1.0 + (p/p_core)**beta_core)

rho_GS = Interp*rho_BPS + (1 - Interp)*rho_Sk
n_GS = np.zeros(len(p))
n_GS[0] = n_Sk[0]
for i in range(1, len(p)):
    n_GS[i] = n_GS[i-1] + (rho_GS[i]-rho_GS[i-1])*n_GS[i-1]/(rho_GS[i-1]+p[i-1])

dInterp = beta_core/p_core**beta_core*p**(beta_core-1.0)/(1.0+(p/p_core)**beta_core)**2
drhodp_GS = dInterp*(rho_BPS - rho_Sk) + Interp + (1.0-Interp)*drhodp_Sk

#------------------------------------------------------------------------------

# Hybrid model (using BCPM)
data_BCPM = np.loadtxt('EoS_Data/EoS_BCPM.dat').T
f = scinter.interp1d(data_BCPM[0], data_BCPM[1])
g = scinter.interp1d(data_BCPM[0], data_BCPM[2])
df = scinter.interp1d(data_BCPM[0], data_BCPM[3])

beta_crust = 2.0
Interp_crust = (p/p_crust)**beta_crust/(1.0 + (p/p_crust)**beta_crust)

rho_HS = Interp_crust*rho_GS + (1 - Interp_crust)*f(p)
n_HS = np.zeros(len(p))
n_HS[0] = g(p[0])
for i in range(1, len(p)):
    n_HS[i] = n_HS[i-1] + (rho_HS[i]-rho_HS[i-1])*n_HS[i-1]/(rho_HS[i-1]+p[i-1])

dInterp_crust = beta_crust/p_crust**beta_crust*p**(beta_crust-1.0)/(1.0+(p/p_crust)**beta_crust)**2
drhodp_HS = dInterp_crust*(rho_GS - f(p)) + Interp_crust*drhodp_GS + (1.0-Interp_crust)*df(p)

#------------------------------------------------------------------------------

# Extract the asymptotic value of lambda2
lambda2 = (rho_GS + p)/(2*np.pi**4*n_GS**2)
lambda2_val = lambda2[-1]

#------------------------------------------------------------------------------

# Save the data
np.savetxt('EoS_Gen.dat', np.array([p, rho_GS, n_GS, drhodp_GS]).T)
np.savetxt('EoS_Hyb.dat', np.array([p, rho_HS, n_HS, drhodp_HS]).T) 