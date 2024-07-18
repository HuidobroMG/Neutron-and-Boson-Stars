"""
@author: huido

Values of useful physical constants and changes of units obtained from the PDG.
"""

import numpy as np

# Fundamental constants
G_SI = 6.67430e-11 # Universal Gravitational constant [N*m2/kg2]
c_SI = 299792458 # Speed of light [m/s]
e_SI = 1.60217662e-19 # Charge of the electron [Coulomb]
me_SI = 9.1093837015e-31 # Electron mass [kg]
mp_SI = 1.67262192369e-27 # Proton mass [kg]
h_SI = 6.62607015e-34 # The Planck constant [J*s]
kB_SI = 1.380649e-23 # Boltzmann constant [J/K]
Da_SI = 1.66053906660e-27 # Dalton mass (a.m.u.) [kg]

# Scale parameters
Msun = 1.98847e30 # Mass of the sun [kg]
Nsun = 1.18855e57 # Baryons in the Sun
nsat = 0.16 # Nuclear saturation baryon density [1/fm3]

# Changes of units
fm_2_km = 1e-18 # Change from fm to km
eV_2_MeV = 1e-6 # Change from eV to MeV
J_2_MeV = eV_2_MeV/e_SI # Change from J to MeV
kg_2_MeV = c_SI**2*J_2_MeV # Change from kg to MeV
MeV_2_Msun = 1.0/(J_2_MeV*c_SI**2*Msun) # Change MeV to Msun
MeV_fm3_2_Msun_km3 = 1.0/(J_2_MeV*c_SI**2*Msun*fm_2_km**3) # Change MeV/fm3 to Msun/km3
MeVfm3_2_Msunkm3 = fm_2_km**3/(J_2_MeV*c_SI**2*Msun) # Change MeV*fm3 to Msun*km3
g_cm3_2_Msun_km3 = 1e12/Msun # Change g/cm3 to Msun/km3
MeV_fm_2_Msun_km = 1.0/(J_2_MeV*c_SI**2*Msun*fm_2_km) # Change MeV/fm to Msun/km

# Useful constants
hbar_SI = h_SI/(2.0*np.pi) # The reduced Planck constant [J*s]
hbarc_MeVfm = 197.3269804 # [MeV*fm]
hbarc_Msunkm = hbarc_MeVfm*MeV_2_Msun*fm_2_km # [Msun*km]
kB_MeV_K = 8.617333262e-11 # Boltzmann Constant [MeV/K]
nsat_km3 = nsat/fm_2_km**3 # Nuclear saturation baryon density [1/km3]
rhosat_MeV_fm3 = nsat*Da_SI*c_SI**2*J_2_MeV
rhosat_Msunkm3 = rhosat_MeV_fm3*g_cm3_2_Msun_km3 # Nuclear saturation energy density [Msun/km3]
kappa_SI = 8.0*np.pi*G_SI/c_SI**4 # Einstein equations constant [s**2/(m*kg)]
kappa_km_Msun = kappa_SI*c_SI**2*Msun*1e-3 # [km/Msun]
G_km_Msun = kappa_km_Msun/(8.0*np.pi) # [km/Msun]

