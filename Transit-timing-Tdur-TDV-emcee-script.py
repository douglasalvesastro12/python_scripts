#from ngts_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pythonFunctions as pfunc
import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
from celerite2.theano import terms, GaussianProcess
import matplotlib.pylab as pylab

save_plots = '/data1/gps_for_ttvs/NOI-106557/ttvs/'
def run_GP(time, flux, flux_err):
    t_model = np.linspace(time.min(), time.max(), int(time.size*5))
    with pm.Model() as model:
        # The mean flux of the time series
        mean = pm.Normal("mean", mu=np.mean(flux), sigma=np.nanstd(flux))

        # A jitter term describing excess white noise
        log_jitter = pm.Normal("log_jitter", mu=np.log(np.nanmedian(flux_err)), sigma=5.0)

        #A term to describe the non-periodic variability
        sigma = pm.InverseGamma(
            "sigma", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
        )
        rho = pm.InverseGamma(
            "rho", **pmx.estimate_inverse_gamma_parameters(0.5, 2.0)
        )

        # The parameters of the RotationTerm kernel
        sigma_rot = pm.InverseGamma(
            "sigma_rot", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
        )
        log_period = pm.Normal("log_period", mu=np.log(per), sigma=np.log(2))
        period = pm.Deterministic("period", tt.exp(log_period))
        log_Q0 = pm.HalfNormal("log_Q0", sigma=2.0)
        log_dQ = pm.Normal("log_dQ", mu=0.0, sigma=2.0)
        f = pm.Uniform("f", lower=0.1, upper=1.0)

        # Set up the Gaussian Process model
       # kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=1/np.sqrt(2.0))
        kernel = terms.RotationTerm(
            sigma=sigma_rot,
            period=period,
            Q0=tt.exp(log_Q0),
            dQ=tt.exp(log_dQ),
            f=f,
        )
        gp = GaussianProcess(
            kernel,
            t=time,
            diag=flux_err**2 + log_jitter**2,
            mean=mean,
            quiet=True,
        )

        # Compute the Gaussian Process likelihood and add it into the
        # the PyMC3 model as a "potential"
        gp.marginal("gp", observed=flux)

        # Compute the mean model prediction for plotting purposes
        pm.Deterministic("pred", gp.predict(flux))

        # Compute the mean model prediction for plotting purposes with different time candences
        pm.Deterministic("pred_t_model", gp.predict(flux, t=t_model))
        #pm.Deterministic("pred_t_transits", gp.predict(flux, t=full_time))

        # Optimize to find the maximum a posteriori parameters
        map_soln = pmx.optimize(verbose=False)

    return model, gp, map_soln, t_model
## ==================================== ##

# Load data
# tess_data_S04 = np.loadtxt('/home/astropc/Universidade/UniversidadChile/research/NOI-106557/data/tess_spoc04_norm.txt').T
# tess_data_S33 = np.loadtxt('/home/astropc/Universidade/UniversidadChile/research/NOI-106557/data/tess_spoc33_norm.txt').T
# tess_data_S34 = np.loadtxt('/home/astropc/Universidade/UniversidadChile/research/NOI-106557/data/tess_spoc34_norm.txt').T
# QLP data
# tess_data_S04 = np.loadtxt('/home/astropc/Universidade/UniversidadChile/research/NOI-106557/data/qlp04_norm.txt').T
# tess_data_S33 = np.loadtxt('/home/astropc/Universidade/UniversidadChile/research/NOI-106557/data/qlp33_norm.txt').T
# tess_data_S33 = np.delete(tess_data_S33, tess_data_S33[1] < 0.8, axis=1)
# tess_data_S34 = np.loadtxt('/home/astropc/Universidade/UniversidadChile/research/NOI-106557/data/qlp34_norm.txt').T
# tess_data_S34 = np.delete(tess_data_S34, tess_data_S34[1] < 0.8, axis=1)
#tess_data_S61 = np.loadtxt('/home/astropc/Universidade/UniversidadChile/research/NOI-106557/data/qlp61_norm.txt').T
#NGTS data
ngts_data = np.loadtxt('/home/astropc/Universidade/UniversidadChile/research/NOI-106557/data/NOI-106557_NGTS.dat').T

tess_data_S04[0] += 2457000
tess_data_S33[0] += 2457000
tess_data_S34[0] += 2457000
tess_data_S61[0] += 2457000

#bin it to 5 min
#tess_data_S61_bin = np.array(pfunc.bin_data(*tess_data_S61, step=30./(60.*24.), avrg='median', normalize=False))
ngts_data = np.array(pfunc.bin_data(*ngts_data, step=5./(60.*24.), avrg='median', normalize=False))

#tess_data = np.concatenate([tess_data_S04, tess_data_S33, tess_data_S34, tess_data_S61_bin], axis=1)
#tess_data = tess_data_S61_bin



# tess_data[-1] = np.ones(tess_data_S61[-1].size) * 0.001 # 0.001 median of tess S34 error (max of the 3 sectors)


# # Plot data
# fig, ax = plt.subplots(1,1, figsize=(8,4))
#
# ax.errorbar(*ngts_data, fmt='C1.', alpha=0.8)
# ax.errorbar(*tess_data, fmt='C0.', alpha=0.8)
# ax.set_xlabel(r'Time (BJD$_{\rm TDB}$)')
# ax.set_ylabel('Norm. Flux')
# ax.minorticks_on()
# plt.tight_layout()


per, t0 = 2.827971, 2459986.408516
per_err, t0_err = 0.00000153, 0.00042659
rp, a, inc = 0.114788,6.921315,83.771997
rpe = 0.0012
#TESS
# oot_tess_data = pfunc.get_oot_lc(*tess_data, per=per, t0=t0, width=0.03)
# median = np.median(oot_tess_data[1])


# time, flux, flux_err = oot_tess_data
# time_all, flux_all, flux_err_all = tess_data


# #NGTS
oot_ngts_data = pfunc.get_oot_lc(*ngts_data, per=per, t0=t0, width=0.020)
median = np.median(oot_ngts_data[1])

time, flux, flux_err = oot_ngts_data
time_all, flux_all, flux_err_all = ngts_data


# In[ ]:


fig, ax = plt.subplots(1,1)

# ax.errorbar((tess_data[0]-0.5*t0)/per %1,(tess_data[1]/median -1)*1e3,tess_data[2]/median, fmt='.')
# ax.errorbar((oot_tess_data[0]-0.5*t0)/per %1,(oot_tess_data[1]/median -1)*1e3,oot_tess_data[2]/median, fmt='.')

# fig, ax = plt.subplots(1,1)

ax.errorbar((ngts_data[0]-0.5*t0)/per %1,(ngts_data[1]/median -1)*1e3,ngts_data[2]/median, fmt='.')
ax.errorbar((oot_ngts_data[0]-0.5*t0)/per %1,(oot_ngts_data[1]/median -1)*1e3,oot_ngts_data[2]/median, fmt='.')
plt.savefig(save_plots + '../folded-ngts.png')

t_model = np.linspace(time.min(), time.max(), int(time.size*5))
prot=0.677
prot_sig = 1
with pm.Model() as model:
    # The mean flux of the time series
    mean = pm.Normal("mean", mu=np.mean(flux), sigma=np.nanstd(flux))

    # A jitter term describing excess white noise
    log_jitter = pm.Normal("log_jitter", mu=np.log(np.nanmedian(flux_err)), sigma=5.0)

    #A term to describe the non-periodic variability
    sigma = pm.InverseGamma(
        "sigma", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
    )
    rho = pm.InverseGamma(
        "rho", **pmx.estimate_inverse_gamma_parameters(0.5, 2.0)
    )

    sigma = pm.Uniform("log_sigma", lower=np.log(1e-06),upper=np.log(1_000_000))

    rho = pm.Uniform("log_rho", lower=np.log(0.001),upper=np.log(1_000))

    # The parameters of the RotationTerm kernel
    sigma_rot = pm.InverseGamma(
        "sigma_rot", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
    )
#     log_period = pm.Normal("log_period", mu=np.log(prot), sigma=np.log(prot_sig))
    log_period = pm.Uniform("log_period", lower=np.log(0.2), upper=np.log(30))
    period = pm.Deterministic("period", tt.exp(log_period))
    log_Q0 = pm.HalfNormal("log_Q0", sigma=2.0)
    log_dQ = pm.Normal("log_dQ", mu=1.0, sigma=2.0)
    f = pm.Uniform("f", lower=0.1, upper=1.0)

    # Set up the Gaussian Process model
    kernel = terms.SHOTerm(sigma=tt.exp(sigma), rho=tt.exp(rho), Q=1/np.sqrt(3.0))
    kernel += terms.RotationTerm(
        sigma=sigma_rot,
        period=period,
        Q0=tt.exp(log_Q0),
        dQ=tt.exp(log_dQ),
        f=f,
    )
    gp = GaussianProcess(
        kernel,
        t=time,
        diag=flux_err**2 + log_jitter**2,
        mean=mean,
        quiet=True,
    )

    # Compute the Gaussian Process likelihood and add it into the
    # the PyMC3 model as a "potential"
    gp.marginal("gp", observed=flux)

    # Compute the mean model prediction for plotting purposes
    pm.Deterministic("pred", gp.predict(flux))

    # Compute the mean model prediction for plotting purposes with different time candences
    pm.Deterministic("pred_t_model", gp.predict(flux, t=t_model))
    pm.Deterministic("pred_t_transits", gp.predict(flux, t=time_all))

    # Optimize to find the maximum a posteriori parameters
    map_soln = pmx.optimize(verbose=False)

    with model:

        trace = pmx.sample(
            tune=500,
            draws=2000,
            start=map_soln,
            cores=40,
            chains=8,
            target_accept=0.9,
            return_inferencedata=True
        )


# MCMC results
results = trace.to_dict()
model_50q = np.median(results['posterior']['pred_t_model'][:, :, :], axis=[0,1])
model_1sig = np.percentile(results['posterior']['pred_t_model'][:, :, :], [16,84], axis=[0,1])
model_2sig = np.percentile(results['posterior']['pred_t_model'][:, :, :], [2.5,97.5], axis=[0,1])
model_3sig = np.percentile(results['posterior']['pred_t_model'][:, :, :], [0.2,99.8], axis=[0,1])
# evaluated on full data
model_50q_full = np.median(results['posterior']['pred_t_transits'][:, :, :], axis=[0,1])

# Get transits TESS
# Tdur = pfunc.transit_dur(per=per, rp=rp, a=a, inc=inc) # in days
# Ti, Te, T0_first = pfunc.linear_ephemerides(T0=t0, T0err=t0_err, P=per, Perr=per_err, Tdur=Tdur, Tdurerr=0.0, dt=0.0, time=tess_data[0])

# Get transits NGTS
Tdur = pfunc.transit_dur(per=per, rp=rp, a=a, inc=inc) # in days
Ti, Te, T0_first_NGTS = pfunc.linear_ephemerides(T0=t0, T0err=t0_err, P=per, Perr=per_err, Tdur=Tdur, Tdurerr=0.0, dt=0.0, time=ngts_data[0])

# Tdur_e = pfunc.transit_dur(per=per+per_err, rp=rp+rpe, a=a, inc=inc) - Tdur


# # Juliet detrended data
# tess_data = np.loadtxt('/home/astropc/Universidade/UniversidadChile/research/NOI-106557/juliet_runs/run2_tess-qlp61_tightPriors_livepoints10k/TESS_GPmodeled.txt',
#                       delimiter=',').T

# # Get transits TESS
# Tdur = pfunc.transit_dur(per=per, rp=rp, a=a, inc=inc) # in days
# Ti, Te, T0_first = pfunc.linear_ephemerides(T0=t0, T0err=t0_err, P=per, Perr=per_err, Tdur=Tdur, Tdurerr=0.0, dt=0.0, time=tess_data[0])


#TESS
# t_transitTESS, f_transitTESS, ferr_transitTESS, NTESS = pfunc.select_full_transits(Ti, Te, per, 2, 100, False, *tess_data, plot=False)

#Linear ephemerides based on t0.
# linear_ephTESS = T0_first + NTESS * per

#NGTS
Ti_NGTS, Te_NGTS, T0_first_NGTS = pfunc.linear_ephemerides(T0=t0, T0err=t0_err, P=per, Perr=per_err, Tdur=Tdur, Tdurerr=0.0, dt=0.0, time=ngts_data[0])
t_transitNGTS, f_transitNGTS, ferr_transitNGTS, NNGTS = pfunc.select_full_transits(Ti_NGTS, Te_NGTS, per, 0., 100, False, *ngts_data, plot=False)

#Linear ephemerides based on t0.
linear_ephNGTS = T0_first_NGTS + NNGTS * per


# fig, ax = plt.subplots(1,1)

# ax.errorbar(*tess_data, fmt='k.')
# ax.plot(linear_ephTESS, [0.975]*linear_ephTESS.size, 'r^')


fig, ax = plt.subplots(1,1)

ax.errorbar(*ngts_data, fmt='k.')
ax.plot(linear_ephNGTS, [0.975]*linear_ephNGTS.size, 'r^')
plt.savefig(save_plots + '../ngts-transitsFlagged.png')

# In[ ]:


# for i in range(NNGTS.size):
#     print(f'T_p1_NGTS_{NNGTS[i]}','\t normal','\t', f'{round(linear_ephNGTS[i],4)},0.05')


#Ploting tools
fig, (ax, ax2) = plt.subplots(2,1, figsize=(10,4), sharex=True, sharey=True, gridspec_kw={'height_ratios':[1, 1]})
#ax.plot(data[0][idx_masked_fluxes]-tmin, (data[1][idx_masked_fluxes]/mu -1)*1e3, color = 'gray', marker='.', linestyle = '', label="outliers")
# ax.errorbar(*tess_data, color = 'black', marker='.', linestyle = '', alpha=0.8, label='data')
ax.errorbar(*ngts_data, color = 'black', marker='.', linestyle = '', alpha=0.8, label='data')
ax.plot(t_model, model_50q, color="red")
ax.fill_between(t_model, *model_1sig, color="red", alpha=0.3)
ax.fill_between(t_model, *model_2sig, color="red", alpha=0.2)
ax.fill_between(t_model, *model_3sig, color="red", alpha=0.1)


ax2.errorbar(ngts_data[0], ngts_data[1] - model_50q_full + 1, ngts_data[2], fmt='k.')
ax2.plot(linear_ephNGTS, [0.975]*linear_ephNGTS.size, 'r^')
# ax2.errorbar(tess_data[0], tess_data[1] - model_50q_full + 1, tess_data[2], fmt='k.')
# ax2.plot(linear_ephTESS, [0.975]*linear_ephTESS.size, 'r^')
#ax2.errorbar(detr_data[0], detr_data[1], detr_data[2], fmt='g.')

offset = 5/(24) # 5hr before/after transit
# for i,j in zip(Ti, Te):
#     ax2.vlines(i-offset,ymin=0, ymax=2, colors='g')
#     ax2.vlines(j+offset,ymin=0, ymax=2, colors='b')

# ax.legend(fontsize=10)
fig.supylabel("Norm. flux")
ax2.set_xlabel("time (days)")
ax.set_ylim(0.97, 1.01)
# ax2.set_ylabel("Norm. flux")
ax.minorticks_on()


plt.tight_layout()
plt.savefig(save_plots + '../ngts-GPmodel-detrended.png')

# Detrended dat
np.savetxt(f'{save_plots}/../detrent_sec61.dat', np.c_[ngts_data[0], ngts_data[1] - model_50q_full + 1, ngts_data[2]], fmt='%s')
np.savetxt(f'{save_plots}/../gpModel.dat', np.c_[t_model, model_50q, model_1sig[0], model_1sig[1], model_2sig[0],model_2sig[1],
                                                          model_3sig[0], model_3sig[1]],
           header='t_model model_50q model_1sig_low model_1sig_up model_2sig_low model_2sig_up model_3sig_low model_3sig_up', fmt='%s')


# np.savetxt('/home/astropc/Desktop/celerit2_detrended_lc.dat', np.c_[tess_data[0], tess_data[1] - model_50q_full + 1, tess_data[2]], fmt='%s')


# for i in range(NTESS.size):
#     print(f'T_p1_TESS_{NTESS[i]}','\t normal','\t', f'{round(linear_ephTESS[i],4)},0.05')


# ## MCMC for each T0


def f_batman(x, t0, rp,a,baseline):
    """
    Function for computing transit models for the set of 8 free paramters
    x - time array
    """
    params = batman.TransitParams()
    params.t0 = t0                     #time of inferior conjunction
    params.per = 2.827971                  #orbital period
    params.rp = rp         #planet radius (in units of stellar radii)
    params.a = a                      #semi-major axis (in units of stellar radii)
    params.inc = 83.771997                     #orbital inclination (in degrees)
    params.ecc = 0.0                     #eccentricity
    params.w = 90.                       #longitude of periastron (in degrees)
    params.u = [0.0092, 0.5254]                #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"       #limb darkening model

    m = batman.TransitModel(params, x)    #initializes model
    flux_m = m.light_curve(params)          #calculates light curve
    return np.array(flux_m)+baseline


# In[ ]:


# t0,per,rp,a,inc,baseline
# y,n,y,y,n,y
priors = np.array([linear_ephNGTS[0]-0.1,linear_ephNGTS[0]+0.1,
          rp-0.05,rp+0.05,
          a-0.5,a+0.5,
           0.0-1e-1, 0.0+1e-1]).reshape(4,2)



# priors = np.array([linear_ephNGTS[0]-0.1,linear_ephNGTS[0]+0.1,
#           rp-0.05,rp+0.05,
#           a-0.5,a+0.5,
#           0.0-1e-1, 0.0+1e-1]).reshape(4,2)


def log_likelihood(theta, x, y, yerr):
#     t0, per, rp, a, inc, ecc, w, baseline = theta
    model = f_batman(x, *theta)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2. * np.pi*sigma2))


def log_prior(theta):
    # The parameters are stored as a vector of values, so unpack them
    t0, rp, a, baseline = theta
    # We're using only uniform priors:
    if priors[0,0] > t0 or t0 > priors[0,1]:
        return -np.inf
    if priors[1,0] > rp or rp > priors[1,1]:
        return -np.inf
    if priors[2,0] > a or a > priors[2,1]:
        return -np.inf
    if priors[3,0] > baseline or baseline > priors[3,1]:
        return -np.inf
    return 0.0

## The full log-probability function is:

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


import emcee
import batman
from multiprocessing import Pool
from IPython.display import display, Math
import corner


# detr_data = np.array([tess_data[0],
#                       tess_data[1] - model_50q_full + 1,
#                       tess_data[-1]
#                      ])


detr_data = np.array([ngts_data[0],
                      ngts_data[1] - model_50q_full + 1,
                      ngts_data[-1]
                     ])
# transit_mask = (detr_data[0] > linear_ephNGTS[np.where(NNGTS == 523)[0][0]] - offset) & (detr_data[0] < linear_ephNGTS[np.where(NNGTS == 523)[0][0]] + offset)



# #Juliet GP detrended data
# detr_data = np.loadtxt('/home/astropc/Universidade/UniversidadChile/research/NOI-106557/juliet_runs/run2_tess-qlp61_tightPriors_livepoints10k/TESS_GPmodeled.txt',
#                       delimiter=',').T

# detr_data = detr_data[detr_data.T[0] < tess_data_S04[0].max() + 100]# Juliet detrended data


folder = 'ngts' #'new_runs/qlp61-with3sigJitter-GProtation'#'juliet-detrent-alldata-withJitter'


# In[ ]:


plt.ioff()
offset = 5/(24) # Get continuum 5hr before/after transit
tess_lc_jitter = 1295.254656*1e-6 #868.8536197200*1e-6#779.5596577824*1e-6 # median sigma_tess #
labels=['t0','rp','a','baseline']

for idx,t0_transit in enumerate(linear_ephNGTS):

    transit_mask = (detr_data[0] > t0_transit - offset) & (detr_data[0] < t0_transit + offset)

    priors = np.array([t0_transit-0.1,t0_transit+0.1,
          rp-0.1,rp+0.1,
          a-1,a+1,
          0.0-1e-1, 0.0+1e-1]).reshape(4,2)

    ndim = priors.shape[0]
    nwalkers = 100

    pos = np.zeros((ndim, nwalkers))
    for n, (c,d) in enumerate(priors):
        pos[n] = (d - c) * np.random.random_sample(nwalkers) + c

    # If you want to use parallelization
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        args=(detr_data[0][transit_mask], detr_data[1][transit_mask],
                                              np.sqrt(detr_data[2][transit_mask]**2 + tess_lc_jitter**2)),
                                        pool=pool)
        # This line below will fit the entire data set, incl OOT. Time consuming.
        # sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(time, flux, flux_error),pool=pool)
        sampler.run_mcmc(pos.T, 10_000, progress=True)


    #plotting
    fig, axes = plt.subplots(ndim, sharex=True, figsize=(9.0, 12.0))

    for i in range(ndim):
        axes[i].plot(sampler.chain[:,:,i].T)
        axes[i].set_ylabel(labels[i])
        axes[i].axhline(y=np.median(priors, axis=1)[i], linestyle='--', lw=1.5, color='k')
    axes[-1].set_xlabel('Step')
    plt.savefig(f'{save_plots}/{folder}/plots/t0_{NNGTS[idx]}_traces.png')
    plt.close('all')

    flat_samples = sampler.get_chain(discard=2_000, thin=200, flat=True)

    fig = corner.corner(flat_samples, labels=labels)
    plt.savefig(f'{save_plots}/{folder}/plots/corner_t0_{NNGTS[idx]}.png')
    plt.close('all')


    params_median = []
    for i in range(ndim):
        params_median.append(np.median(flat_samples[:, i]))
    params_median = np.array(params_median)

    x_model = np.linspace(detr_data[0][transit_mask].min(), detr_data[0][transit_mask].max(), 1_000)
    y_model = f_batman(x_model, *params_median)

    fig, ax = plt.subplots(1,1)
    ax.errorbar(detr_data.T[transit_mask].T[0] - params_median[0], detr_data.T[transit_mask].T[1],
                np.sqrt(detr_data[2][transit_mask]**2 + tess_lc_jitter**2), fmt='.', color='green')
    ax.errorbar(detr_data.T[transit_mask].T[0] - params_median[0], *detr_data.T[transit_mask].T[1:], fmt='.')
    ax.plot(x_model- params_median[0], y_model, 'r--')

    ax.set_xlabel('Time - T0')
    ax.set_ylabel('Norm. Flux')
    for i in np.random.randint(0,flat_samples.shape[0], size=50):
        ax.plot(x_model- params_median[0], f_batman(x_model, *flat_samples[i]), 'r-', alpha=0.2)
    plt.savefig(f'{save_plots}/{folder}/plots/t0_{NNGTS[idx]}_model.png')
    plt.close('all')

    #save_distributions
    np.savetxt(f'{save_plots}/{folder}/t0_{NNGTS[idx]}_sample.txt',
               flat_samples, header='t0 rp a baseline', delimiter='\t', fmt='%s')

    plt.close('all')

plt.ion()


# ## Linear fit to ephemeris

def straight_line(N, T0, PER):
    return PER*N + T0

def log_likelihood_slope(theta, x, y, yerr):
#     t0, per, rp, a, inc, ecc, w, baseline = theta
    model = straight_line(x, *theta)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2. * np.pi*sigma2))


def log_prior_slope(theta):
    # The parameters are stored as a vector of values, so unpack them
    T0, PER = theta
    # We're using only uniform priors:
    if prior_slope[0,0] > T0 or T0 > prior_slope[0,1]:
        return -np.inf
    # inclination must be less than 90
    if prior_slope[1,0] > PER or PER > prior_slope[1,1]:
        return -np.inf
    return 0.0

## The full log-probability function is:

def log_probability_slope(theta, x, y, yerr):
    lp = log_prior_slope(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_slope(theta, x, y, yerr)


# In[ ]:


import os
files = os.listdir(f'{save_plots}/{folder}/')

t0_median, rp_median, a_median, bl_median = [], [], [], []
t0_err, rp_err, a_err, bl_err = [], [], [], []
epochs = []
tdur_median, tdur_err = [], []
for file in files:
    if file.split('.')[-1] == 'txt':
        if file != 'plots':
            epochs.append(int(file.split('_')[1]))
            temp = np.loadtxt(f'{save_plots}/{folder}/{file}')
            t0_median.append(np.median(temp.T[0])); t0_err.append(max(np.diff(np.percentile(temp.T[0], [18, 50, 84]))))
            rp_median.append(np.median(temp.T[1])); rp_err.append(max(np.diff(np.percentile(temp.T[1], [18, 50, 84]))))
            a_median.append(np.median(temp.T[2])); a_err.append(max(np.diff(np.percentile(temp.T[2], [18, 50, 84]))))
            bl_median.append(np.median(temp.T[3])); bl_err.append(max(np.diff(np.percentile(temp.T[3], [18, 50, 84]))))
            temp_tdur = pfunc.transit_dur(per, temp.T[1], temp.T[2], inc)
            tdur_err.append(np.diff(np.percentile(temp_tdur, [18,50,84])).max())
            tdur_median.append(np.median(temp_tdur))

idxs_sort = np.argsort(epochs)
t0_median, t0_err, epochs = np.array(t0_median), np.array(t0_err), np.array(epochs)
#t0_median, t0_err, epochs = np.array(t0_median)[idxs_sort], np.array(t0_err)[idxs_sort], np.array(epochs)[idxs_sort]
#t0_median, t0_err, epochs = (t0_median-np.min(t0_median)), t0_err, epochs-np.min(epochs)


# t0_median, t0_err, epochs = t0_median[(epochs>10) &(epochs<300)], t0_err[(epochs>10) &(epochs<300)], epochs[(epochs>10) &(epochs<300)]

# t0_median, t0_err, epochs = (t0_median-np.min(t0_median)), t0_err, epochs-np.min(epochs)


# fig, ax = plt.subplots(1,1)

# ax.errorbar(epochs, t0, yerr=t0_err, fmt='k.')


prior_slope = [min(t0_median) - 10, min(t0_median) + 10,
              per - 1e-1, per +1e-1]
prior_slope = np.array(prior_slope).reshape(2,2)

nwalkers, ndim = 50, 2
pos = np.zeros((ndim, nwalkers))
for n, (c,d) in enumerate(prior_slope):
    pos[n] = (d - c) * np.random.random_sample(50) + c

# If you want to use parallelization
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_slope, args=(epochs[idxs_sort], t0_median[idxs_sort], t0_err[idxs_sort]),pool=pool)
    # This line below will fit the entire data set, incl OOT. Time consuming.
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(time, flux, flux_error),pool=pool)
    sampler.run_mcmc(pos.T, 10_000, progress=True)


labels=['T0','PER']

try:
    tau = sampler.get_autocorr_time()
    print(tau)
except:
    print('Tau bad!')

fig, axes = plt.subplots(ndim, sharex=True, figsize=(6, 9))

for i in range(ndim):
    axes[i].plot(sampler.chain[:,:,i].T)
    axes[i].set_ylabel(labels[i])
    axes[i].axhline(y=np.median(prior_slope, axis=1)[i], linestyle='--', lw=1.5, color='k')
axes[-1].set_xlabel('Step')
plt.savefig(f'{save_plots}/{folder}/plots/trace_linearfit.png')


flat_samples = sampler.get_chain(discard=2_000, thin=100, flat=True)
fig = corner.corner(flat_samples, labels=labels)
plt.tight_layout()
plt.savefig(f'/{save_plots}/{folder}/plots/corner_linearfit.png')



params_median = []
for i in range(ndim):
    params_median.append(np.median(flat_samples[:, i]))
params_median = np.array(params_median)

x_model = np.linspace(epochs.min(), epochs.max(), 1_000)
y_model = straight_line(x_model, *params_median)



#TESS
# OC = np.zeros(NTESS.size)
# OC_lerr = np.zeros(NTESS.size)
# OC_uerr = np.zeros(NTESS.size)
#NGTS
OC = np.zeros(NNGTS.size)
OC_lerr = np.zeros(NNGTS.size)
OC_uerr = np.zeros(NNGTS.size)
count=0
for file in files:
    if file.split('.')[-1] == 'txt':
        if file != 'plots':
            temp = np.loadtxt(f'{save_plots}/{folder}/{file}')
            obs = temp.T[0]
            computed = flat_samples.T[0] + int(file.split('_')[1])*flat_samples.T[1]
            #straight_line(int(file.split('_')[1]), *flat_samples.T)
            OC[count] =  np.median(obs - computed); OC_lerr[count] = np.diff(np.percentile(obs - computed, [16, 50, 84]))[0]
            OC_uerr[count] = np.diff(np.percentile(obs - computed, [16, 50, 84]))[1]
            count+=1


fig, ax = plt.subplots(1,1, figsize=(10,4))#, gridspec_kw={'width_ratios': [1,3]})

ax.errorbar(epochs[idxs_sort], OC[idxs_sort] * 24 * 60, yerr=np.array([OC_lerr[idxs_sort],OC_uerr[idxs_sort]])*24*60, fmt='ko',
           alpha=0.8,mfc='white', zorder=1)
ax.plot([epochs[idxs_sort].min()-0.1,epochs[idxs_sort].max()+0.1], [0,0], 'k--')
ax.set_xlabel('Transit Number')
ax.set_ylabel('O-C (min)')

plt.tight_layout()

plt.savefig(f'{save_plots}/{folder}/plots/OC.png')
# plt.close('all')

np.savetxt(f'{save_plots}/{folder}/plots/OC.txt',
           np.c_[epochs[idxs_sort], OC[idxs_sort] * 24 * 60, OC_lerr[idxs_sort]*24*60, OC_uerr[idxs_sort]*24*60],
           header='TransiNumber    OC[hr]    OC_1sig-lerr    OC_1sig-uerr', delimiter = '\t', fmt='%s')


fig, ax = plt.subplots(1,1, figsize=(10,4))#, gridspec_kw={'width_ratios': [1,3]})

ax.errorbar(epochs[idxs_sort], OC[idxs_sort] * 24 * 60, yerr=np.array([OC_lerr[idxs_sort],OC_uerr[idxs_sort]])*24*60, fmt='ko',
           alpha=0.8,mfc='white', zorder=1)
ax.plot([epochs[idxs_sort].min()-0.1,epochs[idxs_sort].max()+0.1], [0,0], 'k--')
ax.set_xlabel('Transit Number')
ax.set_ylabel('O-C (min)')

plt.tight_layout()

plt.savefig(f'{save_plots}/{folder}/plots/OC.png')
# plt.close('all')

np.savetxt(f'{save_plots}/{folder}/plots/OC.txt',
           np.c_[epochs[idxs_sort], OC[idxs_sort] * 24 * 60, OC_lerr[idxs_sort]*24*60, OC_uerr[idxs_sort]*24*60],
           header='TransiNumber    OC[hr]    OC_1sig-lerr    OC_1sig-uerr', delimiter = '\t', fmt='%s')



fig, ax = plt.subplots(1,1, figsize=(10,4), gridspec_kw={'width_ratios': [1]})

ax.errorbar(epochs, t0_median, yerr=t0_err, fmt='k.')
ax.plot(x_model, y_model, 'r--')

ax.set_xlabel('Transit Number')
ax.set_ylabel('T0')
for i in np.random.randint(0,flat_samples.shape[0], size=50):
    ax.plot(x_model, straight_line(x_model, *flat_samples[i]), 'r-', alpha=0.2)

# ax2.errorbar(epochs, (t0_median-straight_line(epochs, *flat_samples[i])) * 24 * 60, t0_err*24*60, fmt='ko')
# ax2.set_xlabel('Transit Number')
# ax2.set_ylabel('O-C (min)')

plt.tight_layout()

plt.savefig(f'/{save_plots}/{folder}/plots/linear_fit.png')
# plt.close('all')


fig, (ax, ax2, ax3) = plt.subplots(1,3, figsize=(14,3), sharex=True, sharey=False, gridspec_kw={'width_ratios': [1,1,1]})

ax.errorbar(epochs[idxs_sort], np.array(rp_median)[idxs_sort], yerr=np.array(rp_err)[idxs_sort], fmt='ko')

ax.set_xlabel('Transit Number')
ax.set_ylabel('rp/rs')

ax2.errorbar(epochs, np.array(tdur_median)[idxs_sort] * 24, np.array(tdur_err)[idxs_sort] * 24, fmt='ko')
ax2.set_xlabel('Transit Number')
ax2.set_ylabel('Transit Duration (Hr)')

ax3.errorbar(epochs, np.array(a_median)[idxs_sort], yerr=np.array(a_err)[idxs_sort], fmt='ko')

ax3.set_xlabel('Transit Number')
ax3.set_ylabel('a/rs')

plt.tight_layout()

plt.savefig(f'{save_plots}/{folder}/plots/rp-tdur-a.png')
# plt.close('all')

np.savetxt(f'{save_plots}/{folder}/plots/rp-tdur-a.txt',
           np.c_[epochs[idxs_sort], np.array(rp_median)[idxs_sort],np.array(rp_err)[idxs_sort],
                 np.array(tdur_median)[idxs_sort] * 24, np.array(tdur_err)[idxs_sort] * 24,
                np.array(a_median)[idxs_sort], np.array(a_err)[idxs_sort]],
           header='TransiNumber    rp    rp_1sigerr Tdur[hr] Tdur_err[hr] a a_err', delimiter = '\t', fmt='%s')

