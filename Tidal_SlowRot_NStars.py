"""
@author: HuidobroMG


"""

# Import the modules
import numpy as np
import scipy.integrate as scin
import scipy.interpolate as scinter
import matplotlib.pyplot as plt # Plots
import Units as u

# Rewrites the print statements
cursor_up = '\x1b[2K\r'

# Read the EOS for NS
data = np.loadtxt('BCPM.dat').T
f = scinter.interp1d(data[0], data[1])
g = scinter.interp1d(data[0], data[2])
df = scinter.interp1d(data[0], data[3])

# Define the maximal and minimal pressures
pmax = 0.9*max(data[0]) # [MeV/fm**3]
pmin = 1.1*min(data[0]) # [MeV/fm**3]
pmx = pmax*u.MeV_fm3_2_Msun_km3 # [Msun/km**3]
pmn = pmin*u.MeV_fm3_2_Msun_km3 # [Msun/km**3]

# Physical scales
scalep = pmx/5.0 # [Msun/km**3]
scalew = np.sqrt(u.G_km_Msun*scalep) # [1/km]

# Pressure and rotational frequency values
ip = 1.5
iw = 0.01 # Every magnitud will scale as powers of this parameter

# Space grid
R0 = 1e-4
Rf = 30.0
step = 1e-3
r = np.arange(R0, Rf, step)

# Initial conditions
a0 = 1.0 # Arbitrary
h2p0 = 1.0 # Arbitrary

# Equation of state of the Neutron star
def EoS(p):
    ps = p/u.MeV_fm3_2_Msun_km3
    rho = f(ps)
    n = g(ps)
    drho_dp = df(ps)
    return np.array([rho*u.MeV_fm3_2_Msun_km3, n/u.fm_2_km**3, drho_dp])

# TOV Equations
def TOV(r, var):
    a, M, p = var
    
    if p <= pmn:
        p = 0
        rho = 0
        n = 0
    else:
        rho, n, drho_dp = EoS(p)
    
    if r == 0.0:
        du1 = 0.0
        du2 = 0.0
        B = 1.0
    else:
        du1 = 1.0/r
        du2 = 1.0/(r*(r-2.0*u.G_km_Msun*M))
        B = r**2*du2
    
    A = np.exp(a)

    # Equations
    da = 8*np.pi*u.G_km_Msun*r*B*p + (B-1.0)*du1
        
    dM = 4.0*np.pi*r**2*rho
    
    dp = -(rho + p)*da/2.0
    
    return np.array([da, dM, dp])

# Tidally deformed and Slowly Rotating (Hartle-Thorne) system of equations
def Tidal_HT(r, var):
    a, M, p, w, dw, h2p, k2p, h2h, k2h, m0, pstar, H2, dH2, H2m, dH2m = var
    
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
        B = 1.0
    else:
        du1 = 1.0/r
        du2 = 1.0/(r*(r-2.0*u.G_km_Msun*M))
        B = r**2*du2

    # O = 0
    da = 2.0*(4.0*np.pi*r**3*p + M)*u.G_km_Msun*du2
        
    dM = 4.0*np.pi*r**2*rho
    
    dp = -(rho + p)*da/2.0
    
    # O = 1   
    ddw = -4.0*(1.0-np.pi*r**2*(rho+p)*B*u.G_km_Msun)*du1*dw + 16.0*np.pi*(rho+p)*B*w*u.G_km_Msun
    
    # O = 2 (l = 2)
    #zeta2 = -r**2/B*(3.0*h2p + r**2*np.exp(-a)*w**2)/(3.0*(M*K+4.0*np.pi*p*r**3))
    
    m2p = -r/B*h2p +1.0/6.0*r**4*np.exp(-a)/B*(r/B*dw**2+16.0*np.pi*r*w**2*(rho+p)*u.G_km_Msun)
    
    a1 = (r-3.0*M*u.G_km_Msun-4.0*np.pi*p*u.G_km_Msun*r**3)*B*h2p*du1**2+(r-M*u.G_km_Msun+4.0*np.pi*p*u.G_km_Msun*r**3)*B**2*m2p*du1**3
    a2 = (r-M*u.G_km_Msun+4.0*np.pi*p*u.G_km_Msun*r**3)*B*du1
    a3 = (3.0-4.0*np.pi*(rho+p)*u.G_km_Msun*r**2)*B*h2p*du1+2.0*du1*B*k2p+(1.0+8.0*np.pi*p*u.G_km_Msun*r**2)*m2p*B**2*du1**2+r**3/12.0*np.exp(-a)*dw**2-4.0*np.pi*(rho+p)*r**3*w**2*np.exp(-a)*B/3.0
    
    dh2p = (a3-a1*a2)/(1.0-a2)
    
    dk2p = -dh2p + a1
    
    m2h = -r/B*h2h
    
    b1 = (r-3.0*M*u.G_km_Msun-4.0*np.pi*p*u.G_km_Msun*r**3)*B*h2h*du1**2+(r-M*u.G_km_Msun+4.0*np.pi*p*u.G_km_Msun*r**3)*B**2*m2h*du1**3
    b2 = (r-M*u.G_km_Msun+4.0*np.pi*p*u.G_km_Msun*r**3)*B*du1
    b3 =  (3.0-4.0*np.pi*(rho+p)*u.G_km_Msun*r**2)*B*h2h*du1+2.0*du1*B*k2h+(1.0+8.0*np.pi*p*u.G_km_Msun*r**2)*m2h*B**2*du1**2
    
    dh2h = (b3-b1*b2)/(1.0-b2)
    
    dk2h = -dh2h + b1
    
    # O = 2 (l = 0)
    j = np.exp(-a/2.0)/np.sqrt(B)
    dB = 2.0*u.G_km_Msun*(dM/r - M/r**2)/(1.0-2.0*M*u.G_km_Msun/r)**2
    dj = -np.exp(-a/2.0)/(2.0*np.sqrt(B))*(da + dB/B)
    
    dm0 = 4.0*np.pi*r**2*drho_dp*(rho+p)*u.G_km_Msun*pstar + 1.0/12.0*j**2*r**4*dw**2 - 2.0/3.0*r**3*j*dj*w**2
    
    dpstar = -4.0*np.pi*(p+rho)*u.G_km_Msun*r**2*pstar/(r-2.0*u.G_km_Msun*M) - m0*r**2/(r-2.0*u.G_km_Msun*M)**2*(1.0/r**2+8.0*np.pi*p*u.G_km_Msun) + r**4*j**2/(12.0*(r-2.0*u.G_km_Msun*M))*dw**2 + 1.0/3.0*((r-2.0*u.G_km_Msun*M)*(3.0*r**2*j**2*w**2 + r**3*2.0*j*dj*w**2+2.0*r**3*j**2*w*dw) - r**3*j**2*w**2*(1.0-2.0*dM*u.G_km_Msun))/(r-2.0*u.G_km_Msun*M)**2    

    # Gravito-Electric Deformability    
    ddH2 = -(2.0/r+(2.0*M*u.G_km_Msun*du1**2+4.0*np.pi*r*(p-rho)*u.G_km_Msun)*B)*dH2+(6.0*B*du1**2-4.0*np.pi*u.G_km_Msun*(5*rho+9*p+(rho+p)*drho_dp)*B+da**2)*H2
    
    # Gravito-Magentic Deformability    
    ddH2m = 4.0*np.pi*r*(p+rho)*u.G_km_Msun*B*dH2m + (6.0*r-4.0*M*u.G_km_Msun-8.0*np.pi*r**3*u.G_km_Msun*(p+rho))*B*du1**3*H2m

    return np.array([da, dM, dp, dw, ddw, dh2p, dk2p, dh2h, dk2h, dm0, dpstar, dH2, ddH2, dH2m, ddH2m])


# Adimensional redefinitions of the observables
def Ibar(I, M):
    # [I] = km3
    # [M] = Msun
    Ibar = I/M**3/u.G_km_Msun**3
    return Ibar

def Qbar(Q, J, M):
    # [Q] = km3
    # [J] = km2
    # [M] = Msun
    Qbar = -Q*M*u.G_km_Msun/(J*J)
    return Qbar

def Comp(M, R):
    # [M] = Msun
    # [R] = km
    C = M/R*u.G_km_Msun
    return C

# Electric Tidal Love Number
def k2elec_tidal(y, C):
    f1 = 8.0/5.0*C**5*(1-2.0*C)**2*(2.0+2.0*C*(y-1.0)-y)
    f2 = 2.0*C*(6.0-3.0*y+3.0*C*(5.0*y-8.0))+4.0*C**3*(13.0 - 11.0*y+C*(3.0*y-2.0) + 
        2.0*C*C*(1.0+y))+3.0*(1.0-2.0*C)**2*(2.0 - y+2.0*C*(y-1.0))*np.log(1.0-2.0*C)
    return f1/f2

# Magnetic Tidal Love Number
def k2mag_tidal(y, C):
    f1 = 96*C**5/5.0*(3.0 + 2.0*C*(y-2.0) - y)
    f2 = 2.0*C*(9.0-3.0*y+C*(3.0*(y-1.0)+2.0*C*(C+y+C*y)))+3.0*(3.0+2.0*C*(y-2.0)-y)*np.log(1.0-2.0*C)
    return f1/f2

# Auxiliar functions to compute the Quadrupolar Moment
def C1(M,R,J):
    return (1.0+M*u.G_km_Msun/R)/(u.G_km_Msun*M*R**3)*J**2
def C2(M,R):
    return 3.0*R**2/(M*u.G_km_Msun*(R-2.0*u.G_km_Msun*M))*(1.0-3.0*M*u.G_km_Msun/R+4.0/3.0*(M*u.G_km_Msun/R)**2 + 
            2.0/3.0*(M*u.G_km_Msun/R)**3+R/(2.0*u.G_km_Msun*M)*(1.0-2.0*u.G_km_Msun*M/R)**2*np.log(1.0-2.0*u.G_km_Msun*M/R))
def C3(M,R,J):
    return (1.0+2.0*M*u.G_km_Msun/R)/(u.G_km_Msun*M*R**3)*J**2
def C4(M,R):
    return 3.0*R/(M*u.G_km_Msun)*(1.0+M*u.G_km_Msun/R-2.0/3.0*(M*u.G_km_Msun/R)**2+R/(2.0*M*u.G_km_Msun)*(1.0 - 
            2.0*(M*u.G_km_Msun/R)**2)*np.log(1.0-2.0*M*u.G_km_Msun/R))


def NStar(p0, w0):    
    rho0, n0, drho_dp0 = EoS(p0)
    
    # Initial conditions
    a2 = 4.0*np.pi/3.0*(rho0 + 3.0*p0)*u.G_km_Msun
    a_i = a0 + a2*R0**2
    p2 = -2.0*np.pi/3.0*(rho0 + p0)*(rho0 + 3.0*p0)*u.G_km_Msun
    p_i = p0 + p2*R0**2
    rhoi, ni, drho_dpi = EoS(p_i)
    rho2 = (rhoi - rho0)/R0**2
    M3 = 4.0*np.pi/3.0*rho0
    M5 = 4.0*np.pi/5.0*rho2
    M_i = M3*R0**3 + M5*R0**5
    
    w2 = 8.0*np.pi/5.0*(rho0 + p0)*w0*u.G_km_Msun
    w_i = w0 + w2*R0**2
    dw_i = 2.0*w2*R0
    
    h2p_i = h2p0*R0**2
    k2p_i = -h2p_i
    h2h_i = h2p_i
    k2h_i = -h2h_i
    
    pstar2 = w0**2*np.exp(-a0)/2.0
    m05 = 4*w0**2/15*np.exp(-a0)*(a2 + M3*u.G_km_Msun) - 8*np.pi/5*pstar2*rho2*u.G_km_Msun/a2
    pstar_i = pstar2*R0*R0
    m0_i = m05*R0**5
    
    H2_i = R0**2
    dH2_i = 2.0*R0
    
    H2m_i = R0**3
    dH2m_i = 3.0*R0**2
    
    # Solve the zero order equations to obtain the correct initial conditions
    vinic = [a_i, M_i, p_i]
    sol = scin.solve_ivp(TOV, (R0, Rf), vinic, method = 'RK45',
                        t_eval = r, rtol = 1e-13, atol = 1e-13)
    a, M, p = sol.y

    idx = np.where(p <= pmn)[0][0]
    
    Cv = np.log(1 - 2.0*u.G_km_Msun*M[idx]/r[idx]) - a[idx]
    a0new = a0 + Cv
    
    # Set the correct initial condition for a
    a_i = a0new + a2*R0**2
    pstar2 = w0**2*np.exp(-a0new)/2.0
    m05 = 4.0*w0**2/15.0*np.exp(-a0new)*(a2 + M3*u.G_km_Msun) - 8.0*np.pi/5.0*pstar2*rho2*u.G_km_Msun/a2
    pstar_i = pstar2*R0**2
    m0_i = m05*R0**5
    
    # Solve the tidally deformed slowly rotating stars   
    vinic = [a_i, M_i, p_i, w_i, dw_i, h2p_i, k2p_i, h2h_i, k2h_i, m0_i, pstar_i, H2_i, dH2_i, H2m_i, dH2m_i]

    sol = scin.solve_ivp(Tidal_HT, (R0, Rf), vinic, method = 'RK45',
                        t_eval = r, rtol = 1e-13, atol = 1e-13)
    
    a, M, p, w, dw, h2p, k2p, h2h, k2h, m0, pstar, H2, dH2, H2m, dH2m = sol.y
    
    h0 = r**2*w**2*np.exp(-a)/3 - pstar
    
    idx = np.where(p <= pmn)[0][0]
    rho, n, drho_dp = EoS(p[:idx])
    
    vals = [a[idx], M[idx], p[idx], w[idx], dw[idx], h2p[idx], k2p[idx], h2h[idx], k2h[idx], m0[idx], pstar[idx], H2[idx], dH2[idx], H2m[idx], dH2m[idx]]
    da, dM, dp, dw, ddw, dh2p, dk2p, dh2h, dk2h, dm0, dpstar, dH2, ddH2, dH2m, ddH2m = Tidal_HT(r[idx], vals)
    
    rhoR = rho[-1]
    B = r/(r-2.0*u.G_km_Msun*M)
    
    Radius = r[idx]
    Mass = M[idx]
    Baryon = scin.simpson(4*np.pi*r[:idx]**2*np.sqrt(B[:idx])*n/u.Nsun, x = r[:idx])
    C = Comp(Mass, Radius)
    
    J = Radius**4*dw/6.0 # [km**2]
    Omega = w[idx] + 2.0*J/Radius**3 # [1/km]
    I = J/Omega # [km**3]
    Inertia = Ibar(I,Mass)
    
    yf = Radius*dH2/H2[idx] - 4.0*np.pi*rhoR*Radius**3/Mass # In the abscence of crust
    LambdaT = 2.0/3.0*k2elec_tidal(yf, C)/C**5
    yfm = Radius*dH2m/H2m[idx]
    LambdaTm = abs(1/48.0*k2mag_tidal(yfm,C)/C**5)
    
    c1 = C1(Mass,Radius,J)
    c2 = C2(Mass,Radius)
    c3 = C3(Mass,Radius,J)
    c4 = C4(Mass,Radius)
    CA = 1/(c4+c2*k2h[idx]/h2h[idx])*(c3+k2p[idx]+k2h[idx]/h2h[idx]*(c1-h2p[idx]))
    CB = 1/h2h[idx]*(c1 - CA*c2-h2p[idx])
    Q = -J**2/(Mass*u.G_km_Msun) - 8.0/5.0*CA*(Mass*u.G_km_Msun)**3 # [km**3]
    Quadrupolar = Qbar(Q,J,Mass)
    
    LambdaR = -Q/(Omega**2*(Mass*u.G_km_Msun)**5)
    
    deltaM = (m0[idx] + J**2/Radius**3 + 4.0*np.pi*Radius**3/Mass*(Radius-2.0*u.G_km_Msun*Mass)*rhoR*pstar[idx])/u.G_km_Msun
    
    h0i = J**2/(Radius**3*(Radius-2.0*Mass*u.G_km_Msun)) - deltaM*u.G_km_Msun/(Radius-2.0*Mass*u.G_km_Msun) - h0[idx]
    
    h2 = h2p + CB*h2h
    k2 = k2p + CB*k2h
    h0 += h0i

    mp0 = 4.0*np.pi*rho*r[:idx]**2*np.sqrt(B[:idx])
    mp2 = mp0*(2.0*r[:idx]**2*m0[:idx]/(r[:idx] - 2.0*u.G_km_Msun*M[:idx]) + 1/3.0*r[:idx]**4*w[:idx]**2*np.exp(-a[:idx]/2.0))
    
    Mp0 = scin.simpson(mp0, x = r[:idx])
    Mp2 = scin.simpson(mp2, x = r[:idx])
    
    return [np.array([a, M, p, w, h2, k2, H2, H2m]),
            np.array([Radius, Mass, Baryon, J, Inertia, Quadrupolar, LambdaR, LambdaT, LambdaTm, deltaM, Mp0, Mp2])]


# Solve one case
p0 = ip*scalep
w0 = iw*scalew
sols = NStar(p0, w0)
print(sols[1])

'''
# Solve many cases
p0 = scalep*np.exp(np.linspace(np.log(8e-3), np.log(5.0), 50))
w0 = iw*scalew

Nstars = len(p0)
Mass = np.zeros(Nstars)
Radius = np.zeros(Nstars)
Baryon = np.zeros(Nstars)
J = np.zeros(Nstars)
Inertia = np.zeros(Nstars)
Quadrupolar = np.zeros(Nstars)
LambdaR = np.zeros(Nstars)
LambdaT = np.zeros(Nstars)
LambdaTm = np.zeros(Nstars)
DeltaM = np.zeros(Nstars)
Mp0 = np.zeros(Nstars)
Mp2 = np.zeros(Nstars)

for i in range(Nstars):
    print(cursor_up+str(i+1)+'/'+str(Nstars), end = '\r')
    sols = NStar(p0[i], w0)
    R, Ms, B, AngM, I, Q, LR, LT, LTm, dM, mp0, mp2 = sols[1]
    Radius[i] = R
    Mass[i] = Ms
    Baryon[i] = B
    J[i] = AngM
    Inertia[i] = I
    Quadrupolar[i] = Q
    LambdaR[i] = LR
    LambdaT[i] = LT
    LambdaTm[i] = abs(LTm)
    DeltaM[i] = dM
    Mp0[i] = mp0
    Mp2[i] = mp2


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

# Check the YY relations
fig = plt.figure(figsize = (12, 8))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.plot(LambdaT, Inertia, 'b.', label = 'Results')
ax1.plot(LambdaT, YY_Fit(LambdaT, pars_IL), 'r-', label = 'YY Fit')

ax2.plot(LambdaT, Quadrupolar, 'b.')
ax2.plot(LambdaT, YY_Fit(LambdaT, pars_QL), 'r-')

ax3.plot(Quadrupolar, Inertia, 'b.')
ax3.plot(Quadrupolar, YY_Fit(Quadrupolar, pars_IQ), 'r-')

ax4.plot(LambdaT, LambdaR, 'b.')
ax4.plot(LambdaT, YY_Fit(LambdaT, pars_LL), 'r-')

ax1.set_xlabel(r'$\lambda_t$', fontsize = 15)
ax1.set_ylabel(r'$\bar{I}$', fontsize = 15)
ax2.set_xlabel(r'$\lambda_t$', fontsize = 15)
ax2.set_ylabel(r'$\bar{Q}$', fontsize = 15)
ax3.set_xlabel(r'$\bar{Q}$', fontsize = 15)
ax3.set_ylabel(r'$\bar{I}$', fontsize = 15)
ax4.set_xlabel(r'$\lambda_t$', fontsize = 15)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax4.set_xscale('log')
ax4.set_yscale('log')

ax1.legend(fontsize = 10)

plt.show()
'''