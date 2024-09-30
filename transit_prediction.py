import numpy as np
from astropy.time import Time
import batman
import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation
from astropy import units as u

def f_batman(x, t0, a, per=3.690621, rp=0.0396, inc=89.3, baseline=0.0, ecc=0.0,w=90, u = [0.45, 0.45] ,limb_dark ="quadratic"):
    """
    Function for computing transit models for the set of 8 free paramters
    x - time array
    """
    params = batman.TransitParams()
    params.t0 = t0                     #time of inferior conjunction
    params.per = per                  #orbital period
    params.rp =  rp         #planet radius (in units of stellar radii)
    params.a = a                      #semi-major axis (in units of stellar radii)
    params.inc = inc                     #orbital inclination (in degrees)
    params.ecc = ecc                     #eccentricity
    params.w = w                       #longitude of periastron (in degrees)
    params.u = u               #limb darkening coefficients [u1, u2]
    params.limb_dark = limb_dark       #limb darkening model

    m = batman.TransitModel(params, x)    #initializes model
    flux_m = m.light_curve(params)          #calculates light curve
    return np.array(flux_m)+baseline

def transit_dur(per,rp,a,inc):
    '''
    Compute transit duration from eq. ? from Haswell book transiting exoplanet.
    pars: a, rp, per is parameterised as batmam package
    '''

    b = a * np.cos(inc * np.pi/180)
    tdur = (per/np.pi) * np.arcsin( np.sqrt( (1 + rp)**2 - b**2) / a ) #for e=0

    #IMPLEMENT ERROR PROPAGATION HERE
    #https://physics.stackexchange.com/questions/503748/error-propagation-in-sine
    #If enter with distributions, error propagation is not necessary

    return tdur

def transit_window(T0, T0err, P, Perr, Tdur, Tdurerr, N):
    '''
    Compute transit ingress and egress from given parameters
    dt: add a fractional oot wings. e.g, dt=0.2 means 20% of Tdur
    '''
    Ti = T0 + N*P - (N*Perr + T0err + 0.5 * Tdur + Tdurerr) #Ti = Beginning of window
    Te = T0 + N*P + (N*Perr + T0err + 0.5 * Tdur + Tdurerr) #Te (late) = End of transit window

    return (Ti, Te, T0+N*P)

def transit_prediction():
    """
    Enter with Gregorian date in the format yyyy-mm-dd hh:mm:ss
    """
    return

def next_transits(T0, T0err, P, Perr, Tdur, Tdurerr, N, transits=True, quadratures=True):
    """
    N: number of transits to display. N=10, the following 10 transits in Gregorian date and UT
    """
    observing_location = EarthLocation(lat=-29.2563*u.deg, lon=-70.7380*u.deg, height=2400*u.m)
    time_t0=Time(f'{T0}',format='jd')
    time_t0=time_t0.iso
    if transits:
        print(f'Next {N} transit events in UTC from T0 at: {time_t0}')
        for n in range(0,N+1):
            #Tn,Tn_err = T0 + n*P, T0err + n*Perr
            ti,te,t0 = transit_window(T0, T0err, P, Perr, Tdur, Tdurerr, n)
            time_ti=Time(f'{ti}',format='jd', location=observing_location)
            time_ti=time_ti.iso

            time_t0=Time(f'{t0}',format='jd', location=observing_location)
            time_t0=time_t0.iso

            time_te=Time(f'{te}',format='jd', location=observing_location)
            time_te=time_te.iso
            print('Ti -- Tmid -- Te: ', time_ti, '--', time_t0,'--', time_te)

    if quadratures:
        for n in range(0,N+1):
            #Tn,Tn_err = T0 + n*P, T0err + n*Perr
            ti,te,t0 = transit_window(T0, T0err, P, Perr, Tdur, Tdurerr, n)
            phases = [t0 - P/4, t0, t0+P/4, t0+P/2]

            phases_quadrature1=Time(f'{phases[0]}',format='jd', location=observing_location).iso
            phases_quadrature2=Time(f'{phases[1]}',format='jd', location=observing_location).iso
            phases_quadrature3=Time(f'{phases[2]}',format='jd', location=observing_location).iso
            phases_quadrature4=Time(f'{phases[3]}',format='jd', location=observing_location).iso

            print('phi: -0.25, 0.0, +0.25, +0.5', phases_quadrature1, '--', phases_quadrature2,'--', phases_quadrature3,
                  '--', phases_quadrature4)


def compute_time_at_rv_phase():
    """
    Returns the time to get the rv point at a given radial velocity phase. Phase is assumed from 0 to 1. Transits are around 0.5. E.g, Quadrature points at highest RVs 0.25 and 0.75. Enters with pars + rv_phase --> time
    """
    return

# from PyAstronomy import pyasl
# Compute Keplerian for a given orbital settings
# T0, T0err = 2459986.4084634269, 0.00035
# P, Perr = 2.8279693976, 0.0000012
# rp,a,inc = 0.1163463184,6.7631588457,83.6359586226 # a is a/rstar
# rs, ecc = 1.50, 0.0
# rsun_meters = 6.957e+8
# a_meters, per_sec = a * rs * (rsun_meters), P * 24 * 60 * 60
# start_time, end_time, sampling_time = T0,T0 + 10*P,100
# t = np.linspace(start_time, end_time, sampling_time) * 24 * 60 * 60
# #longitude of ascending node, inclination, periapsis argument --> Omega, i, w
# ke = pyasl.KeplerEllipse(a_meters, per_sec, e=ecc, Omega=90., i=inc, w=0.0) # m/s
# vel = ke.xyzVel(t)
#
# plt.plot(t, vel[::, 2], 'r.-')
# plt.show()



# fake transits pars
# T0, T0err = 2459986.4084634269, 0.0#, 0.00035
# P, Perr = 2.8279693976,0.0# 0.0000012
# rp,a,inc = 0.1163463184,6.7631588457,83.6359586226
# Tdur = 0/(60*24)#transit_dur(P,rp,a,inc)
# Tdurerr = 0.0#0.5/24

# Make lightcurve
# x_time = np.arange(-10.,30, 5/(60*24)) + T0
# y_model = f_batman(x_time, t0=T0, a=a, per=P, rp=rp, inc=inc, baseline=0.0, ecc=0.0,w=90, u = [0.45, 0.45] ,limb_dark ="quadratic")
# w_noise = 0.0 + 0.001*np.random.randn(x_time.size)
# y = y_model + w_noise
#
# plt.plot(x_time, y, '.')
# plt.plot(x_time, y_model, 'r-')
# plt.show()

#

# TOI-6442
# T0, T0err = 2459986.40838, 0.00055
# P, Perr = 2.827969,0.000002
# rp,a,inc = 0.1157,6.91,83.78
# Tdur = 2.60/24#transit_dur(P,rp,a,inc)
# Tdurerr = 0.01/24#0.1/24 # from Tdur plot

# Matt NOI-106475 / TIC
# T0, T0err = 2458765.6218, 0.0014
# P, Perr = 1.143404, 0.000020
# Tdur = 0.0720#transit_dur(P,rp,a,inc)
# Tdurerr = 0.0022 # from Tdur plot
#

# TOI-333
T0, T0err = 2458355.5712790191, 0.0033763121
P, Perr = 3.7852571765, 0.0000074138
rp,a,inc = 0.0340074939, 9.2047573432, 89.2488309595
Tdur = transit_dur(P,rp,a,inc)
Tdurerr = 0.1/24 # from Tdur plot

# # TOI-2607
# T0, T0err = 2459256.9815070503, 0.0822925465
# P, Perr = 2.2344903022, 0.0093226897
# # rp,a,inc = 0.0178, 9.4141777415, 89.0985830826
# Tdur = 2/24#transit_dur(P,rp,a,inc)
# Tdurerr = 0.1/24 # from Tdur plot

# TOI-2607c
# T0, T0err = 2516840.261, 0.0822925465
# P, Perr = 17.40636778, 0.00
# # rp,a,inc = 0.0178, 9.4141777415, 89.0985830826
# Tdur = 2/24#transit_dur(P,rp,a,inc)
# Tdurerr = 0.1/24 # from Tdur plot

next_transits(T0, T0err, P, Perr, Tdur, Tdurerr, N=600, transits=True, quadratures=False)






# NOI-106557 UTC (for local chilean time -3hr) transit times --> phase = 0
# Ti -- Tmid -- Te:  2024-01-04 21:26:01.315 -- 2024-01-04 22:52:11.731 -- 2024-01-05 00:18:22.147
# Ti -- Tmid -- Te:  2024-01-07 17:18:17.767 -- 2024-01-07 18:44:28.287 -- 2024-01-07 20:10:38.806
# Ti -- Tmid -- Te:  2024-01-10 13:10:34.219 -- 2024-01-10 14:36:44.842 -- 2024-01-10 16:02:55.466
# Ti -- Tmid -- Te:  2024-01-13 09:02:50.671 -- 2024-01-13 10:29:01.398 -- 2024-01-13 11:55:12.125
# Quadrature points at phi: -0.25, 0.0, +0.25, +0.5
#2024-01-04 05:54:07.592 -- 2024-01-04 22:52:11.731 -- 2024-01-05 15:50:15.870 -- 2024-01-06 08:48:20.009
#2024-01-07 01:46:24.148 -- 2024-01-07 18:44:28.287 -- 2024-01-08 11:42:32.426 -- 2024-01-09 04:40:36.565
#2024-01-09 21:38:40.703 -- 2024-01-10 14:36:44.842 -- 2024-01-11 07:34:48.981 -- 2024-01-12 00:32:53.120
#2024-01-12 17:30:57.259 -- 2024-01-13 10:29:01.398 -- 2024-01-14 03:27:05.537 -- 2024-01-14 20:25:09.676

# TOI-333 UTC (for local chilean time -3hr) transit times --> phase = 0
