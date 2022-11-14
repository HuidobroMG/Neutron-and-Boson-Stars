# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:02:52 2022

@author: huido
"""

# PACKAGES

import numpy as np
import scipy.integrate as si # RK integration
import scipy.special as se # For Model 1
import scipy.interpolate as scinter # For Nuclear Physics EoS
import matplotlib.pyplot as plt # Plots
import csv # This helps us to write the data in a file
import os # To obtain our path
import Units

#-----------------------------------------------------------------------------

name = 'BCPM'
data = np.loadtxt('EoS_Data/EoS_'+name+'.dat').T

f = scinter.interp1d(data[0], data[1])
g = scinter.interp1d(data[0], data[2])
df = scinter.interp1d(data[0], data[3])

pmax = 0.9*max(data[0])
pmin = 1.1*min(data[0])
pmx = pmax*Units.MeV_fm3_2_Msun_km3
pmn = pmin*Units.MeV_fm3_2_Msun_km3

p_PT = 0 # MeV/fm3
quadratic_interp = False

if p_PT != 0:
    dataB = np.loadtxt('EoS_Data/BCPM.dat').T
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

# MANY STARS
NumStars = 0

#-----------------------------------------------------------------------------

# EQUATION OF STATE
def EoS(p):
    # p is the pressure
    p /= Units.MeV_fm3_2_Msun_km3
    
    rho = f(p)
    n = g(p)
    drho_dp = df(p)
    if p <= p_PT:
        alpha = (p/p_PT)**2/(1 + (p/p_PT)**2)
        dalpha = 2*p/p_PT**2/(1 + (p/p_PT)**2)**2
        rho = (1-alpha)*fB(p) + alpha*f(p)
        n = n_func(p)
        drho_dp = dalpha*(f(p) - fB(p)) + (1-alpha)*dfB(p) + alpha*df(p)

    return np.array([rho*Units.MeV_fm3_2_Msun_km3, n/Units.fm_2_km**3, drho_dp])

#-----------------------------------------------------------------------------

# TOV Equations
def TOV(r, var):
    
    nu, M, p, N = var
    
    if p <= pmn:
        p = 0
        rho = 0
        n = 0
    else:
        rho, n, drho_dp = EoS(p)
    
    if r == 0.0:
        du5 = 0.0
        B = 1.0
    else:
        du5 = 1.0/(r*(r-2.0*Units.K*M))
        B = r**2*du5

    # O = 0
    dnu = 2.0*(4.0*np.pi*r**3*p + M)*Units.K*du5
        
    dM = 4.0*np.pi*r**2*rho
    
    dp = -(rho + p)*dnu/2.0
    
    dN = 4.0*np.pi*r**2*n*np.sqrt(B)/Units.Nsun
    
    return np.array([dnu, dM, dp, dN])

def TOV_Rota(r, var):
    
    nu, M, p, N, w, W, h2p, k2p, h2h, k2h, m0, pstar, H2, Beta, H2m, Gamma = var
    
    if p <= pmn:
        p = 0
        rho = 0
        n = 0
        drho_dp = 0
    else:
        rho, n, drho_dp = EoS(p)
    
    if r == 0.0:
        du1 = 0.0
        du2 = 0.0
        du3 = 0.0
        du4 = 0.0
        du5 = 0.0
        B = 1.0
    else:
        du1 = 1.0/r
        du2 = 1.0/r**2
        du3 = 1.0/r**3
        du4 = 1.0/r**4
        du5 = 1.0/(r*(r-2.0*Units.K*M))
        B = r**2*du5

    # O = 0
    dnu = 2.0*(4.0*np.pi*r**3*p + M)*Units.K*du5
        
    dM = 4.0*np.pi*r**2*rho
    
    dp = -(rho + p)*dnu/2.0
    
    dN = 4.0*np.pi*r**2*n*np.sqrt(B)/Units.Nsun
    
    # O = 1
    dw = W
    
    ddw = -4.0*(1.0-np.pi*r**2*(rho+p)*B*Units.K)*du1*dw + 16.0*np.pi*(rho+p)*B*w*Units.K
    
    # O = 2 (L = 2)
    #zeta2 = -r**2/B*(3.0*h2p + r**2*np.exp(-nu)*w**2)/(3.0*(M*K+4.0*np.pi*p*r**3))
    
    m2p = -r/B*h2p +1.0/6.0*r**4*np.exp(-nu)/B*(r/B*dw**2+16.0*np.pi*r*w**2*(rho+p)*Units.K)
    
    a1 = (r-3.0*M*Units.K-4.0*np.pi*p*Units.K*r**3)*B*h2p*du2+(r-M*Units.K+4.0*np.pi*p*Units.K*r**3)*B**2*m2p*du3
    a2 = (r-M*Units.K+4.0*np.pi*p*Units.K*r**3)*B*du1
    a3 = (3.0-4.0*np.pi*(rho+p)*Units.K*r**2)*B*h2p*du1+2.0*du1*B*k2p+(1.0+8.0*np.pi*p*Units.K*r**2)*m2p*B**2*du2+r**3/12.0*np.exp(-nu)*dw**2-4.0*np.pi*(rho+p)*r**3*w**2*np.exp(-nu)*B/3.0
    
    dh2p = (a3-a1*a2)/(1.0-a2)
    
    dk2p = -dh2p + a1
    
    m2h = -r/B*h2h
    
    b1 = (r-3.0*M*Units.K-4.0*np.pi*p*Units.K*r**3)*B*h2h*du2+(r-M*Units.K+4.0*np.pi*p*Units.K*r**3)*B**2*m2h*du3
    b2 = (r-M*Units.K+4.0*np.pi*p*Units.K*r**3)*B*du1
    b3 =  (3.0-4.0*np.pi*(rho+p)*Units.K*r**2)*B*h2h*du1+2.0*du1*B*k2h+(1.0+8.0*np.pi*p*Units.K*r**2)*m2h*B**2*du2
    
    dh2h = (b3-b1*b2)/(1.0-b2)
    
    dk2h = -dh2h + b1
    
    # O = 2 (L = 0)
    j = np.exp(-nu/2.0)/np.sqrt(B)
    dB = 2.0*Units.K*(dM/r - M/r**2)/(1.0-2.0*M*Units.K/r)**2
    dj = -np.exp(-nu/2.0)/(2.0*np.sqrt(B))*(dnu + dB/B)
    
    dm0 = 4.0*np.pi*r**2*drho_dp*(rho+p)*Units.K*pstar + 1.0/12.0*j**2*r**4*dw**2 - 2.0/3.0*r**3*j*dj*w**2
    
    dpstar = -4.0*np.pi*(p+rho)*Units.K*r**2*pstar/(r-2.0*Units.K*M) - m0*r**2/(r-2.0*Units.K*M)**2*(1.0/r**2+8.0*np.pi*p*Units.K) + r**4*j**2/(12.0*(r-2.0*Units.K*M))*dw**2 + 1.0/3.0*((r-2.0*Units.K*M)*(3.0*r**2*j**2*w**2 + r**3*2.0*j*dj*w**2+2.0*r**3*j**2*w*dw) - r**3*j**2*w**2*(1.0-2.0*dM*Units.K))/(r-2.0*Units.K*M)**2    

    # GRAVITO-ELECTRIC DEFORMABILITY
    dH2 = Beta
    
    dBeta = -(2.0/r+(2.0*M*Units.K*du2+4.0*np.pi*r*(p-rho)*Units.K)*B)*dH2+(6.0*B*du2-4.0*np.pi*Units.K*(5.0*rho+9.0*p+(rho+p)*drho_dp)*B+dnu**2)*H2
    
    # GRAVITO-MAGNETIC DEFORMABILITY
    dH2m = Gamma
    
    dGamma = 4.0*np.pi*r*(p+rho)*Units.K*B*dH2m + (6.0*r-4.0*M*Units.K-8.0*np.pi*r**3*Units.K*(p+rho))*B*du3*H2m

    return np.array([dnu, dM, dp, dN, dw, ddw, dh2p, dk2p, dh2h, dk2h, dm0, dpstar, dH2, dBeta, dH2m, dGamma])


#-----------------------------------------------------------------------------

# Adimensional parameters to plot the I-Love-Q relations 
def Ibar(I, M):
    # [I] = km3
    # [M] = Msun
    Ibar = I/M**3/Units.K**3
    return Ibar

def Qbar(Q, J, M):
    # [Q] = km3
    # [J] = km2
    # [M] = Msun
    Qbar = -Q*M*Units.K/(J*J)
    return Qbar

def Comp(M, R):
    # [M] = Msun
    # [R] = km
    C = M/R*Units.K
    return C

# Function to calculate the electric deformability
def k2elec_tidal(y, C):
    dummy = 8.0/5.0*C**5*(1.0-2.0*C)**2*(2.0+2.0*C*(y-1.0)-y)
    dum = 2.0*C*(6.0-3.0*y+3.0*C*(5.0*y-8.0))+4.0*C**3*(13.0 - 
                11.0*y+C*(3.0*y-2.0)+2.0*C*C*(1.0+y))+3.0*(1.0-2.0*C)**2*(2.0 - 
                         y+2.0*C*(y-1.0))*np.log(1.0-2.0*C)
    return dummy/dum

# Function to calculate the magnetic deformability
def k2mag_tidal(y, C):
    dummy = 96*C**5/5.0*(3.0 + 2.0*C*(y-2.0) - y)
    dum = 2.0*C*(9.0-3.0*y+C*(3.0*(y-1.0)+2.0*C*(C+y+C*y)))+3.0*(3.0+2.0*C*(y-2.0)-y)*np.log(1.0-2.0*C)
    return dummy/dum

# Constants to obtain the Quadrupolar moment
def C1(M,R,J):
    return (1.0+M*Units.K/R)/(Units.K*M*R**3)*J**2
def C2(M,R):
    return 3.0*R**2/(M*Units.K*(R-2.0*Units.K*M))*(1.0-3.0*M*Units.K/R+4.0/3.0*(M*Units.K/R)**2 + 
                    2.0/3.0*(M*Units.K/R)**3+R/(2.0*Units.K*M)*(1.0-2.0*Units.K*M/R)**2*np.log(1.0-2.0*Units.K*M/R))
def C3(M,R,J):
    return (1.0+2.0*M*Units.K/R)/(Units.K*M*R**3)*J**2
def C4(M,R):
    return 3.0*R/(M*Units.K)*(1.0+M*Units.K/R-2.0/3.0*(M*Units.K/R)**2+R/(2.0*M*Units.K)*(1.0 - 
                  2.0*(M*Units.K/R)**2)*np.log(1.0-2.0*M*Units.K/R))

#------------------------------------------------------------------------------

#Initial conditions
R0 = 1e-3
Rf = 30.0
step = 1e-3

r = np.arange(R0, Rf, step)

scalep = 73.8335*Units.MeV_fm3_2_Msun_km3
scalew = np.sqrt(Units.K*scalep)

nu0 = 1.0 # Arbitrary
h2p0 = 1.0 # Arbitrary

ip = 1.0
iw = 0.01 # Every magnitud will scale as powers of this parameter

#------------------------------------------------------------------------------

def Star(p0):   
    w0 = iw*scalew
    
    rho0, n0, drho_dp0 = EoS(p0)
    
    # Initial conditions expanded
    nu2 = 4.0*np.pi/3.0*(rho0+3.0*p0)*Units.K
    nui = nu0 + nu2*R0**2
    p2 = -2.0*np.pi/3.0*(rho0+p0)*(rho0+3.0*p0)*Units.K
    pi = p0 + p2*R0**2
    rhoi, ni, drho_dpi = EoS(pi)
    rho2 = (rhoi-rho0)/R0**2
    M3 = 4.0*np.pi/3.0*rho0
    M5 = 4.0*np.pi/5.0*rho2
    Mi = M3*R0**3 + M5*R0**5
    Ni = 4.0*pi*R0*R0*ni/Units.Nsun
    
    w2 = 8.0*np.pi/5.0*(rho0+p0)*w0*Units.K
    wi = w0+w2*R0**2
    Wi = 2.0*w2*R0
    
    h2pi = h2p0*R0**2
    k2pi = -h2pi
    h2hi = h2pi
    k2hi = -h2hi
    
    pstar2 = w0**2*np.exp(-nu0)/2.0
    m05 = 4.0*w0**2/15.0*np.exp(-nu0)*(nu2 + M3*Units.K) - 8.0*np.pi/5.0*pstar2*rho2*Units.K/nu2
    pstari = pstar2*R0*R0
    m0i = m05*R0**5
    
    H2i = R0**2
    Betai = 2.0*R0
    
    H2mi = R0**3
    Gammai = 3.0*R0**2
    
    #--------------------------------------------------------------------------
    
    # Solve the zero order equations to obtain the correct initial conditions
    vinic = [nui, Mi, pi, Ni]
    sol = si.solve_ivp(TOV, (R0, Rf), vinic, method = 'BDF', t_eval = r,
                       rtol = 1e-12)
    nu, M, p, N = sol.y
    rad = sol.t

    if np.any(p <= 0):
        idx = np.where(p < 1e-15)[0][0]
        if p[idx] <= 0:
            idx -= 1
    else:
        idx = np.where(p == min(p))[0][0] 
    #idx = np.where(p < 1e-15)[0][0]
    #if p[idx] <= 0:
        #idx -= 1
    
    Cv = np.log(1.0-2.0*Units.K*M[idx]/rad[idx]) - nu[idx]
    nu0new = nu0 + Cv
    
    # Set the correct initial condition for nu
    nui = nu0new + nu2*R0**2
    pstar2 = w0**2*np.exp(-nu0new)/2.0
    m05 = 4.0*w0**2/15.0*np.exp(-nu0new)*(nu2 + M3*Units.K) - 8.0*np.pi/5.0*pstar2*rho2*Units.K/nu2
    pstari = pstar2*R0*R0
    m0i = m05*R0**5
    
    #--------------------------------------------------------------------------
    
    # Solve the slowly rotating stars   
    vinic = [nui, Mi, pi, Ni, wi, Wi, h2pi, k2pi, h2hi, k2hi, m0i, pstari, H2i, Betai, H2mi, Gammai]

    sol = si.solve_ivp(TOV_Rota, (R0, Rf), vinic, method = 'BDF', t_eval = r,
                       rtol = 1e-12)
    
    nu, M, p, N, w, W, h2p, k2p, h2h, k2h, m0, pstar, H2, Beta, H2m, Gamma = sol.y
    rad = sol.t
    
    dnu = 2.0*(4.0*np.pi*r**3*p + M)*Units.K/(r*(r-2.0*Units.K*M))
    
    h0 = 1.0/3.0*r*r*w*w*np.exp(-nu) - pstar
    zeta0 = 2.0*pstar*2.0*(4.0*np.pi*r**3*p + M)*Units.K/(r*(r-2.0*Units.K*M))
    
    if np.any(p <= 0):
        idx = np.where(p < 1e-15)[0][0]
        if p[idx] <= 0:
            idx -= 1
    else:
        idx = np.where(p == min(p))[0][0]  
    #idx = np.where(p < 1e-15)[0][0]
    #if p[idx] <= 0:
        #idx -= 1
    
    vals = [nu[idx], M[idx], p[idx], N[idx], w[idx], W[idx], h2p[idx], k2p[idx], h2h[idx], k2h[idx], m0[idx], pstar[idx], H2[idx], Beta[idx], H2m[idx], Gamma[idx]]
    dnu, dM, dp, dN, dw, ddw, dh2p, dk2p, dh2h, dk2h, dm0, dpstar, dH2, dBeta, dH2m, dGamma = TOV_Rota(r[idx], vals)
    
    rho = np.zeros(len(p))
    for i in range(len(p)):
        if i > idx:
            rho[i] = 0
        else:
            rho[i] = EoS(p[i])[0]
    rhoR = rho[idx]
    
    #--------------------------------------------------------------------------
    
    Radius = rad[idx]
    Mass = M[idx]
    Baryon = N[idx]
    C = Comp(Mass, Radius)
    
    J = Radius**4*dw/6.0 # km2
    Omega = w[idx] + 2.0*J/Radius**3 # 1/km
    I = J/Omega # km3
    Inertia = Ibar(I,Mass)
    
    yf = Radius*dH2/H2[idx] - 4.0*np.pi*rhoR*Radius**3/Mass # Ausence of crust
    LambdaT = 2.0/3.0*k2elec_tidal(yf, C)/C**5
    yfm = Radius*dH2m/H2m[idx]
    SigmaT = abs(1.0/48.0*k2mag_tidal(yfm,C)/C**5)
    
    c1 = C1(Mass,Radius,J)
    c2 = C2(Mass,Radius)
    c3 = C3(Mass,Radius,J)
    c4 = C4(Mass,Radius)
    A = 1.0/(c4+c2*k2h[idx]/h2h[idx])*(c3+k2p[idx]+k2h[idx]/h2h[idx]*(c1-h2p[idx]))
    CQ = 1.0/h2h[idx]*(c1 - A*c2-h2p[idx])
    Q = -J**2/(Mass*Units.K) - 8.0/5.0*A*(Mass*Units.K)**3 # km3
    Quad = Qbar(Q,J,Mass)
    
    LambdaR = -Q/(Omega**2*(Mass*Units.K)**5)
    
    deltaM = (m0[idx] + J**2/Radius**3 + 4.0*np.pi*Radius**3/Mass*(Radius-2.0*Units.K*Mass)*rhoR*pstar[idx])/Units.K
    
    h0i = J**2/(Radius**3*(Radius-2.0*Mass*Units.K)) - deltaM*Units.K/(Radius-2.0*Mass*Units.K) - h0[idx]
    
    h0 = h0i + h0

    B = 1.0/(1.0-2.0*M*Units.K/rad)
    mp0 = 4.0*np.pi*rho*r*r*np.sqrt(B)
    mp2 = mp0*(2.0*r*r*m0/(r-2.0*Units.K*M) + 1.0/3.0*r**4*w*w*np.exp(-nu/2.0))
    
    Mp0 = si.simps(mp0[:idx+1], rad[:idx+1])
    Mp2 = si.simps(mp2[:idx+1], rad[:idx+1])
    
    return Radius, Mass, Baryon, J, Inertia, Quad, LambdaT, LambdaR, SigmaT, deltaM, Mp0, Mp2

if NumStars == 0:
    p0 = scalep*np.exp(np.linspace(np.log(pmx/400), np.log(pmx), 25))
    
    Mass = np.zeros(len(p0))
    Radius = np.zeros(len(p0))
    Baryon = np.zeros(len(p0))
    Inertia = np.zeros(len(p0))
    DefT = np.zeros(len(p0))
    Quad = np.zeros(len(p0))
    DefR = np.zeros(len(p0))
    DefM = np.zeros(len(p0))
    DeltaM = np.zeros(len(p0))
    Mp0 = np.zeros(len(p0))
    Mp2 = np.zeros(len(p0))
    J = np.zeros(len(p0))
    
    i = 0
    Nstars = len(p0)
    for p in p0:
        R, Ms, Nb, AngM, I, Q, LT, LR, ST, dM, mp0, mp2 = Star(p)
        Radius[i] = R
        Mass[i] = Ms
        Baryon[i] = Nb
        Inertia[i] = I
        DefT[i] = LT
        Quad[i] = Q
        DefR[i] = LR
        DefM[i] = abs(ST)
        DeltaM[i] = dM
        Mp0[i] = mp0
        Mp2[i] = mp2
        J[i] = AngM
        i += 1
        print(str(i)+'/'+str(Nstars))
else:
    p0 = scalep*ip
    Radius, Mass, Baryon, J, Inertia, Quad, DefT, DefR, DefM, DeltaM, Mp0, Mp2 = Star(p0)


if NumStars == 0:
    # Yagi-Yunes Fit
    pars_IL = np.array([1.49824523e+00,  6.29220764e-02,  2.13905178e-02, -6.02928764e-04,
                        5.39279032e-06])
    
    pars_QL = np.array([1.66976995e-01,  9.02458190e-02,  4.43199090e-02, -3.63939508e-03,
                        9.61802991e-05])
    
    pars_IQ = np.array([1.38889279,  0.65339533, -0.0601616 ,  0.05776263, -0.00523084])
    
    pars_LL = np.array([3.16346745e+00,  2.16089970e-01,  8.71009449e-02, -4.84525264e-03,
                        1.06965881e-04])
    
    def YY_Fit(x, pars):
        log_x = np.log(x)
        return np.exp(pars[0] + pars[1]*log_x + pars[2]*log_x**2 + pars[3]*log_x**3 + pars[4]*log_x**4)
    
    fig = plt.figure(figsize = (12, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    ax1.plot(DefT, YY_Fit(DefT, pars_IL), 'k.')
    ax1.plot(DefT, Inertia, 'b-')
    
    ax2.plot(DefT, YY_Fit(DefT, pars_QL), 'k.')
    ax2.plot(DefT, Quad, 'b-')
    
    ax3.plot(Quad, YY_Fit(Quad, pars_IQ), 'k.')
    ax3.plot(Quad, Inertia, 'b-')
    
    ax4.plot(DefT, YY_Fit(DefT, pars_LL), 'k.')
    ax4.plot(DefT, DefR, 'b-')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax4.set_xscale('log')
    ax4.set_yscale('log')