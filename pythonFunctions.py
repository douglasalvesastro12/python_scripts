import numpy as np
import batman
import collections
import matplotlib.pyplot as plt
import emcee
from multiprocessing import Pool
from IPython.display import display, Math
import corner
from astropy.io import fits
from scipy.optimize import minimize,basinhopping,curve_fit
from astropy.table import Table
#from photutils.datasets import make_gaussian_sources_image
#from photutils.aperture import ApertureStats, CircularAperture
from matplotlib.gridspec import GridSpec

#Compilation of several functions
#=======================================================================
def bootstrapping(t,f,ferr, model, guess, nsampling):
    '''
    Do bootstraping for a given model with curve_fit
    model: model to fit the data
    guess: list input guess pars for model
    nsamplig: number of boostraping realisation
    '''
    pars = []
    for _ in range(nsampling):
        new_data_idx = np.random.choice(np.arange(t.size), size=t.size)
        t_new,f_new,ferr_new = t[new_data_idx],f[new_data_idx],ferr[new_data_idx]

        # Sinusoidal models
        popt, pcov = curve_fit(model, xdata=t_new, ydata=f_new,
                       p0=guess, sigma=ferr_new, absolute_sigma=True)
        pars.append(popt)

    return np.array(pars).T
#=======================================================================
def find_LSpeak(permin, permax, frequency, power):
    #Find max peak given a range of periods
    cond = (permin<1/frequency) & (permax>1/frequency) #gives the period
    idx_maxper = np.argmax(power[cond])
    return 1/frequency[cond][idx_maxper], power[cond].max()
#===============================================
def sigma_clipping(x, y,yerr, nsig, niter, mask_id=False):
    """
    Perform sigma clipping.

    Parameters
    ----------
    x,y,yerr : data, yerr does not enter the sigma clipping.
    nsig : number of sigmas to be clipped
    niter : integer, number of interactions to be performed.
    mask_id: if True return the indexes of clipped points and original dataset

    Note
    -----
    If mu, sigma are the same as previous iteraction the code will stop at that current iteraction
    not reaching the end of niter.

    Returns
    -------
    clipped x,y,yerr data

    """
    original_x = np.copy(x)
    original_y = np.copy(y)
    original_yerr = np.copy(yerr)

    removed_x = []

    count=1
    while count <= niter:
        sig = np.nanstd(y)
        mu = np.nanmedian(y)
        l_bound = (y > (mu - nsig*sig))
        u_bound = (y < (mu + nsig*sig))

        removed_x.append(x[~(l_bound & u_bound)])
        x = x[l_bound & u_bound]
        y = y[l_bound & u_bound]
        yerr = yerr[l_bound & u_bound]

        if (mu <= np.nanmedian(y)) & (sig <= np.nanstd(y)):
            #print(f'stopped at niter:{count}')
            if mask_id:
                removed_x = np.sort(np.concatenate(removed_x)) #removed timepoints
                idxs = np.array([np.where(original_x==i)[0][0] for i in removed_x]) #indexes of the removed timepoints
                data_original = np.array([original_x, original_y, original_yerr])
                return x, y, yerr, idxs, data_original

            return x, y, yerr

        if count == niter:
            if (mu != np.nanmedian(y)) & (sig != np.nanstd(y)):
                print(f'Number of iteraction not enough, ninter incresed to {2*niter}')
                niter*=2
        count+=1
#====================================================

def check_sep(ra1, dec1, ra2, dec2):
    '''
    Compute angular separation between two or more ra, dec and a target star ra,dec.
    ra, dec are in degree
    return: ang_sep, difference between ra's, and difference between dec's
    '''

    from astropy.coordinates import angular_separation

    ang_sep = angular_separation(ra1, dec1, ra2, dec2)

    return ang_sep, ra_all-ra2, dec_all-dec2

#================================================================
def bin_data(t, flux, flux_err, step, avrg='mean', normalize=False):
    '''
    Data binning by mean otherwise median. If one datapoint inside binsize, remove it.
    Parameters:
    t: x axis
    flux: y axis
    step: size of bins
    avrg = How data will be averaged (e.g, mean or median)
    
    '''
    #bin the data, t_binned contains the boarders of bins
    t_binned = np.arange(t.min(), t.max() + step, step) #endpoint not included
    binned_time = [] #time to be plotted against avr flux within a bin
    binned_flux = []
    binned_flux_err = []
    for i in range(len(t_binned)):
        try:
            cond = (t >= t_binned[i]) & (t < t_binned[i+1])
        except IndexError:
            continue
        if len(flux[cond]) > 1:
            binned_time.append(np.median(t[cond])) #bin flux and put it in center of bins may put point in wrong time location dt ~ step/2 which is the case window is at a gap or edge
            binned_flux.append(np.mean(flux[cond]) if avrg == 'mean' else np.median(flux[cond]))
#            binned_flux_err.append(np.sqrt((flux_err[cond]**2).mean()))
            binned_flux_err.append(np.sqrt(np.var(flux[cond])/len(flux[cond]))) #Std error of the mean

        # elif len(flux[cond]) == 1: #if one datapoint, that value is taken
        #     binned_time.append(t[cond][0]) #t[cond] returns an array with 1 element [0] gets the element
        #     binned_flux.append(flux[cond][0])
        #     binned_flux_err.append(flux_err[cond][0])

    if normalize:
        return np.array(binned_time), np.array(binned_flux)/np.nanmedian(binned_flux), np.array(binned_flux_err)/np.nanmedian(binned_flux)
            
    #step/2 makes the data point in the center of the bin
    return np.array(binned_time), np.array(binned_flux), np.array(binned_flux_err)
#========================================================================================
def phase_fold(time, period, epoch):
    # phase fold data:
    phase = (time - epoch) % period / period
    phase[np.where(phase>0.5)] -= 1
    
    return phase
#==========================================================================================
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
#====================================================================================================
def get_oot_lc(time, flux, flux_err, per, t0, width=.035):
    phases = (time - t0)/per %1
    phases[np.where(phases>0.5)] -= 1
    idx_oot = np.where(np.abs(phases)>width)[0]
    return time[idx_oot], flux[idx_oot], flux_err[idx_oot]
#=============================================================
def linear_ephemerides(T0, T0err, P, Perr, Tdur, Tdurerr, dt, time):
    '''
    Compute expected transit ingress and egress from entire given time
    dt: add a fractional oot wings. e.g, dt=0.2 means 20% of Tdur 
    Notice that it assumes continuous data, hence for ground-based some caught transits are empty
    '''
    init_time, end_time = time.min(), time.max()# time.min(), time.max()

    # Check if T0 is the 1st transit. If not shift T0 N transits backwards 
    N_init = int((T0 - init_time) / P) #TODO Ideally it should be T0 - 1st T0. If 1st T0 - init_time is close to 1 Period it may lead to errors...
    if N_init > 0: #TODO works only if t0 is ahead, need to account for t0 before, need print shifting onwards....
        print(f'Shifting T0 backwards {N_init} transits')
        T0 -= N_init * P
    elif N_init < 0:
        print(f'Shifting T0 onwards {np.abs(N_init)} transits')
        T0 += np.abs(N_init) * P
    #Sanity checks
    assert int((T0 - init_time) / P) == 0
    
    dt *= Tdur
    #TODO if data has continuity in the edges that stitched together would be longer than 1 Period, this line will add 1 extra transit number 
    #Ideally it should be last Tc - 1st Tc 
    N = np.arange( int((end_time - init_time) / P) +1) 
    Ti = T0 + N*P - (N*Perr + T0err + 0.5 * Tdur + Tdurerr) - dt #Ti = Beginning of window
    Te = T0 + N*P + (N*Perr + T0err + 0.5 * Tdur + Tdurerr) + dt #Te (late) = End of transit window
    
    return (Ti, Te, T0)
#====================================================================================================
def transit_difference(T0, T0err, T0ref, T0ref_err, P=0.7920520, Perr=0.0000093):
    '''Computes Difference between two T0 central transits. Negative values mean T0ref is ahead (larger) of T0'''
    N = int(abs(T0 - T0ref) / P) #number of transits between T0 and T0ref
    if T0 > T0ref:
        diff = (T0 - (T0ref + N*P)) * 24 * 60 # Difference between T0s in minutes
        diff_err = (T0ref_err + T0err) * 24 * 60 #T0s errors are added to make T0 diff larger. diff_err = T0max - T0
    else:
        #eq [(T0ref- eT0ref) + NP - (T0+eT0)] - (T0ref + NP) - T0 = -eT0ref - eT0 
        diff = ((T0 + N*P) - T0ref) * 24 * 60 
        diff_err = -(T0ref_err + T0err) * 24 * 60
        
    return diff, diff_err
#============================================================================================================
def select_full_transits(Ti, Te, P, npoints, Ntransits, random_transits, *data, plot=True):
    '''
    Select all or a sample Ntransits of transits from LC for when npoints are within transit Ti,Te window.
    Need implementation for when Ntransits > Total transit. Do not exceed this constraint! Make a better while
    loop
    Parameters:
    Ti, Te: from linear_ephemerides function
    P: Planet period [days]
    npoints: Number of data points within transit windown
    random_transits: if True returns random transits
    Ntransits: Number of transits to get from LC. Nonsense if random_transit is False
    data: t, f, ferr 
    '''
    time, flux, flux_err = data
    N = np.arange( int((time.max() - time.min())/P) + 1) #total possible N values
    t, f, ferr = {}, {}, {} #transits is stored here
    N_counts = [] 
    iterations = 0
    if random_transits == True:
        while len(N_counts) < Ntransits:
            if Ntransits > len(N):
                print('Ntransits to catch larger than total N transits in data')
                print(f'Total N transits from data: {len(N)}')
                break
                
            Nrand = np.random.choice(N) #pick transit number from all possible values 
            cond = (time >= Ti[Nrand]) & (time <= Te[Nrand])
            if len(time[cond]) == 0: # if data is discontinuous there may be no data for a given N then
                Nrand = np.random.choice(N) #pick a new N
                iterations +=1
            else:
                if len(time[cond]) >= npoints: #Check there's enough data points within transit windown (Ti, Te)
                    if len(N_counts) != 0:#Do not allow same transit windown to be picked 
                        if Nrand not in N_counts:
                            N_counts.append(Nrand)
                            t[f'transit {Nrand}'] = time[cond]
                            f[f'transit {Nrand}'] = flux[cond]
                            ferr[f'transit {Nrand}'] = flux_err[cond]
                        else:
                            Nrand = np.random.choice(N)
                            
                    else:#1st picked transit will be added here
                        N_counts.append(Nrand)
                        t[f'transit {Nrand}'] = time[cond]
                        f[f'transit {Nrand}'] = flux[cond]
                        ferr[f'transit {Nrand}'] = flux_err[cond]
                        Nrand = np.random.choice(N)
                else:# If 
                    Nrand = np.random.choice(N)
                    iterations +=1
                    if iterations== 1e4:
                        print(f'while loop iterations exceeded: value {iterations}. Investigate!')
                        print('Hints: No transits meet npoints criterium or Ntransits >>> Total N transits from data')
                        break
                
    else:
        N_delete = [] # if empty space in data or npoints is not met else is triggered. Delete indexes
        for ind in N:
            cond = (time >= Ti[ind]) & (time <= Te[ind])
            if (len(time[cond]) > npoints) & (len(time[cond]) != 0.): #make sure at least n datapoints are within the window 
                t[f'transit {ind}'] = time[cond]
                f[f'transit {ind}'] = flux[cond]
                ferr[f'transit {ind}'] = flux_err[cond]
            else:
                N_delete.append(ind)
    
        N = np.delete(N, N_delete)
    
    #unfold each array within dicts to put it into arrays
    transits_t, transits_f, transits_ferr = [], [], []
    #sort t dictionary
#    t = collections.OrderedDict(sorted(t.items()))
    #sort time to return sorted transits
    for i in sorted( N_counts if random_transits else N ): #N_counts if random_transits, if not N is used for catching all dips
        i = 'transit ' + str(i)
        for j,k,z in zip(t[f'{i}'], f[f'{i}'],ferr[f'{i}']):
            transits_t.append(j), transits_f.append(k), transits_ferr.append(z)
            
    N_ = (np.array(N_counts) if random_transits else N)
    if plot:
        fig, ax = plt.subplots(len(N_),1, figsize = (6,len(N_)*2.5))
        for idx, ind in enumerate(N_): #do look in N_counts or N depending on random_transit value
            if len(N_) != 1:
                ax[idx].errorbar(t[f'transit {ind}'], f[f'transit {ind}'], ferr[f'transit {ind}'], fmt='g.', label = f'Transit {ind}')
                ax[idx].legend(loc='best')
            else:
                ax.errorbar(t[f'transit {ind}'], f[f'transit {ind}'], ferr[f'transit {ind}'], fmt='g.', label = f'Transit {ind}')
                ax.legend(loc='best')
        if len(N_) != 1:
            ax[-1].set_xlabel('t0 [days]')
        else:
            ax.set_xlabel('t0 [days]')
        plt.tight_layout()
#Return transits in a single dataset array
#    return np.array(transits_t),np.array(transits_f),np.array(transits_ferr), N_
#return dictionary with individual transits separated
    return t, f, ferr, N_
#===================================================================================================================================
def draw_lc(time,cadence, sigma, pars, plot=True):
    '''
    Draw a sample LC from a normal distribution with mean = model, sdv = sigma.
    Parameters:
    time: initial and final time. Array-like. Cadence defines the time domain. 
    Cadence: in minutes  
    '''
    cadence /= (60 * 24)
    time = np.arange(time[0], time[-1], cadence)
    model = f_batman(time, *pars)
    
    flux, flux_err = np.random.normal(loc=model, scale = sigma), np.array([sigma] * model.size)
    if plot:
        fig, ax = plt.subplots(1,1, figsize = (9,6))

        ax.errorbar(time, flux, flux_err, fmt = 'k.', alpha = 0.1, label = 'mock data')
        ax.plot(time, model, 'r--', label = 'mock data model')
        ax.set_xlabel('Time [days]')
        ax.set_ylabel('Rel. Flux')
        ax.set_title(f'True pars: t0:{pars[0]:.4f}, per:{pars[1]:.4f}, rp:{pars[2]:.4f}, a:{pars[3]:.4f}, inc:{pars[4]:.4f}, bl:{pars[5]:.7f}')
        #, ecc:{ecc:.4f}, w:{w:.4f}, [u1, u2]: {u1, u2}')
        plt.legend()
    
    return time, flux, flux_err
#======================================================================================================================
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
#=====================================================================================================================
def get_transit(Ti, Te, time, flux, flux_err):
    '''
    Pick individual transits from data within windown Ti (time of ingress) and Te (time of egress).
    '''
    
    cond = (time > Ti) & (time < Te)
    
    return time[cond], flux[cond], flux_err[cond] 
#=====================================================================================================================
def get_epochs(Ti, Te, N, time, flux, flux_err):
    '''
    Get minimum value of a selected transit as t0.
    Ti, Te: transit windows
    N: Transit epoch (integer)
    '''
    
    #get epochs
    epochs = []
    for i in N:

        t, f, ferr = get_transit(Ti[i], Te[i], time, flux, flux_err)
        epochs.append( t[f == f.min()][0] )

    epochs = np.array(epochs).flatten()
    
    return epochs
#=======================================================================================
def rms(x):
    '''
    Compute root-mean-square of a variable x. If x has nans it alerts to it and compute rms excluding nans
    '''
    if np.any(np.isnan(x)):
        print('NaN detected. RMS is computed without nans')
        cond = np.isnan(x)
        RMS = ((1/len(x[~cond]))*np.sum(x[~cond]**2))**.5
    else:
        RMS = ((1/len(x))*np.sum(x**2))**.5
    return RMS

#======================================================================================
def NGTS_bynight(t,f,ferr):
    '''
    Enter t, f, ferr timeseries and return a dictionary with the nights. 
    Check condition that individual points cannot be farther from 12h apart, if so they belong 
    to distinct nights.
    '''
    
    j=0
    nights = {}
    indv_night = []

    for i in range(len(t)):
        if i+1 == len(t):
            break
        elif t[i+1] - t[i] < 12/(24): #smaller than 12hr
            indv_night.append([t[i],f[i],ferr[i]])
        else:
            indv_night.append([t[i],f[i],ferr[i]]) #append the last point of the night
            nights[f'night {j}'] = np.copy(indv_night)
            j+=1
            indv_night = []
    
    #append last night
    indv_night.append([t[i],f[i],ferr[i]]) #append the last point of the night
    nights[f'night {j}'] = np.copy(indv_night)
    
    return nights
#===============================================================================
def ngts_binned(t,f,ferr, mode='median', step=30/(60*24), clip_large_errors=False):
    '''
    Split timeseries by night, bin nights individually to step min, remove data points which have errorbars
    larger than 3-sigma from median, and return the entire timeseries to perform the rotation period study.
    mode: median subtracted flux or normalized flux.
    file_data must be an array with time, flux, flux_err in that order.
    returs: t,f,ferr binned and number of clipped large errors if True
    '''
    
    # if type(file_data) == str:
    #     #Load data
    #     t, f, ferr = np.genfromtxt(file_data)
    # else:
    #     t,f,ferr = np.copy(file_data) #otherwise file_data gets accessed by f variable then modified

    if mode=='median':
        f_median = np.median(f)
        f -= f_median
    elif mode=='normalized':
        f_median = np.median(f)
        f /= f_median
        ferr /= f_median

    nights = NGTS_bynight(t,f,ferr)
    #Binning
    t_night, f_night, ferr_night = [], [], []
    for label in nights.keys():
        tbin, fbin, ferrbin = bin_data(*nights[label].T, step=step, avrg='median')
        cond = np.isnan(fbin) | np.isnan(ferrbin)
        tbin, fbin, ferrbin = tbin[~cond], fbin[~cond], ferrbin[~cond]
        t_night.append(tbin), f_night.append(fbin), ferr_night.append(ferrbin)

    tbin = np.concatenate(t_night)
    fbin = np.concatenate(f_night)
    ferrbin = np.concatenate(ferr_night)

    if clip_large_errors == True:
        #Data clipping outliers
        f_q50, f_q18, f_q84 = np.percentile(ferrbin, [50, 0.15, 99.85]) #3-sigma
        #~step min Data to be searched for Prot
        cond = (ferrbin > f_q18) & (ferrbin < f_q84)
        tbin = tbin[cond]
        fbin = fbin[cond]
        ferrbin = ferrbin[cond]

        return tbin, fbin, ferrbin, np.where(~cond)[0].size
            
    return tbin, fbin, ferrbin

def run_tls(t,f, period_min, period_max,path,name,use_threads,binsize=0.005,plot=True,save_plots=False, remove_alias_freq=True):

    '''
    run TLS on t,f data
    '''

    from transitleastsquares import transitleastsquares
    try:
        model = transitleastsquares(t,f)

        results = model.power(period_min=period_min, period_max=period_max,use_threads=use_threads)
        #print('Period', format(results.period, '.5f'), 'd')
        #print(len(results.transit_times), 'transit times in time series:',
            #['{0:0.5f}'.format(i) for i in results.transit_times])
        #print('Transit depth', format(results.depth, '.5f'))
        #print('Best duration (days)', format(results.duration, '.5f'))
        #print('Signal detection efficiency (SDE):', results.SDE)
        if not remove_alias_freq:

            if plot or save_plots:
                fig, (ax,ax2,ax3) = plt.subplots(3,1, figsize=(10,7))
                ax.axvline(results.period, alpha=0.4, lw=3, label = f'Period {results.period:.5f} d')
                ax.set_title(f'P: {results.period:.4f} d, Tdepth: {results.depth:.4f}, Tdur: {24*results.duration:.4f} hr, SDE: {results.SDE:.4f} ')
                ax.set_xlim(np.min(results.periods), np.max(results.periods))
                for n in range(2, 10):
                    ax.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
                    ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
                ax.set_ylabel(r'SDE')
                ax.set_xlabel('Period (days)')
                ax.plot(results.periods, results.power, color='black', lw=0.5)
                ax.set_xlim(results.period-5, results.period+5)

                #x2
                ax2.plot(results.model_folded_phase, results.model_folded_model, color='red')
                ax2.scatter(results.folded_phase, results.folded_y, color='blue', s=10, alpha=0.5, zorder=2)
                t_binned, f_binned, ferr_binned = bin_data(results.folded_phase,
                                                        results.folded_y,np.ones(results.folded_y.size),step=binsize)

                ax2.errorbar(t_binned, f_binned, ferr_binned, fmt='k.')
                ax2.set_ylim(np.median(f_binned) - 5*np.std(f_binned), np.median(f_binned) + 5*np.std(f_binned))

                # ax2.set_ylim(f_binned.min()-0.05, f_binned.max()+0.05)

                # plt.xlim(0.48, 0.52)
                # ax2.set_ticklabel_format(useOffset=False)
                ax2.set_xlabel('Phase')
                ax2.set_ylabel('Relative flux');

                from transitleastsquares import transit_mask

                in_transit = transit_mask(
                    t,
                    results.period,
                    results.duration,
                    results.T0)
                ax3.scatter(
                    t[in_transit],
                    f[in_transit],
                    color='red',
                    s=2,
                    zorder=0)
                ax3.scatter(
                    t[~in_transit],
                    f[~in_transit],
                    color='blue',
                    alpha=0.5,
                    s=2,
                    zorder=0)
                ax3.plot(
                    results.model_lightcurve_time,
                    results.model_lightcurve_model, alpha=0.5, color='red', zorder=1)
                ax3.set_xlim(min(t), max(t))
                ax3.set_xlabel('Time (days)')
                ax3.set_ylabel('Relative flux');
                if save_plots:
                    plt.savefig(path + name + '.png')
                    plt.close(fig)
            return results.period, results.T0, results.SDE, results.depth, results.duration
        else:
            mask_alias = [0.5, 1.0, 2.0, 3.0, 4.0]
            delta = 0.05
            mask_cond = [False] * results.periods.size
            for i in mask_alias:
                mask_cond += (results.periods <= i+delta) & (results.periods >= i-delta) # True + False = True
            results.periods, results.power = results.periods[~mask_cond], results.power[~mask_cond]
            plot_5largest_peak(t,f,results.periods, results.power, path=path, name=name)
            return np.nan, np.nan, np.nan, np.nan
    except:
        return np.nan, np.nan, np.nan, np.nan
#============
def plot_5largest_peak(t,f,periods, powers,path=None, name=None):
    """
    Plot the fase folded timeseries for the 5 largest periods with highest power
    """
    grid = np.arange(periods.min(), periods.max(),0.1)
    max_per_power = []
    binsize = 0.01
    for i in range(1,len(grid)):
        max_per, max_power = find_LSpeak(grid[i-1], grid[i], 1/periods, powers)
        max_per_power.append([max_per, max_power])

    max_per_power_idx = np.argsort(np.array(max_per_power).T[-1])[::-1]
    max_per_power = np.array(max_per_power)[max_per_power_idx]
    max_periods = max_per_power[:10]
    fig = plt.figure(figsize=(10,20))
    row, col = 11, 2
    gs = GridSpec(row, col, figure=fig, height_ratios=[3] + [2]*(row-1))

    ax1 = fig.add_subplot(gs[0, :])
    #per1
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    #per2
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    #per3
    ax6 = fig.add_subplot(gs[3, 0])
    ax7 = fig.add_subplot(gs[3, 1])
    #per4
    ax8 = fig.add_subplot(gs[4, 0])
    ax9 = fig.add_subplot(gs[4, 1])
    #per4
    ax10 = fig.add_subplot(gs[5, 0])
    ax11 = fig.add_subplot(gs[5, 1])
    #per4
    ax12 = fig.add_subplot(gs[6, 0])
    ax13 = fig.add_subplot(gs[6, 1])
    #per4
    ax14 = fig.add_subplot(gs[7, 0])
    ax15 = fig.add_subplot(gs[7, 1])
    #per4
    ax16 = fig.add_subplot(gs[8, 0])
    ax17 = fig.add_subplot(gs[8, 1])
    #per4
    ax18 = fig.add_subplot(gs[9, 0])
    ax19 = fig.add_subplot(gs[9, 1])
    #per4
    ax20 = fig.add_subplot(gs[10, 0])
    ax21 = fig.add_subplot(gs[10, 1])

    ax1.set_xlim(np.min(periods), np.max(periods))
    ax1.set_xscale('log')
    ax1.set_ylabel(r'SDE')
    ax1.set_xlabel('Period (days)')
    ax1.plot(periods, powers, color='black', lw=0.5)
    ax1.minorticks_on()

    for max_per,power in max_periods:
        ax1.plot(max_per,power, 'rx')

    q10, q90 = np.nanpercentile(f, [10, 90])
    for idx,(ax,ax_) in enumerate(zip([ax2, ax4, ax6, ax8, ax10, ax12, ax14, ax16, ax18, ax20],
                                        [ax3, ax5, ax7, ax9, ax11, ax13, ax15, ax17, ax19, ax21])):

        phased = phase_fold(t, max_periods[idx][0], 0.)
        phased += 0.5
        ax.scatter(phased,f, color='blue', s=10, alpha=0.2)
        t_binned, f_binned, ferr_binned = bin_data(phased, f,
                                                   np.ones(f.size),step=binsize)
        ax.errorbar(t_binned, f_binned, ferr_binned, fmt='k.')
        ax.set_ylim(q10, q90)
        ax.minorticks_on()
        ax.set_xlabel('Phase')
        ax.set_ylabel('Relative flux');
        ax.annotate(f'Period = {max_periods[idx][0]:.2f} d', xy=(0.01, 0.01), xycoords='axes fraction', color='black',fontsize=15)

        # Plot zoomed in version
        ax_.scatter(phased,f, color='blue', s=10, alpha=0.2)
        ax_.errorbar(t_binned, f_binned, ferr_binned, fmt='k.')
        ax_.set_ylim(q10, q90)
        ax_.set_xlabel('Phase')
        ax_.set_xlim(0.4,0.6)
        ax_.set_yticks([])
        ax_.minorticks_on()
        ax_.annotate(f'Period = {max_periods[idx][0]:.2f} d', xy=(0.01, 0.01), xycoords='axes fraction', color='black',fontsize=15)
        ax_.annotate(f'ZOOM', xy=(0.75, 0.8), xycoords='axes fraction', color='black',fontsize=15)
        plt.tight_layout()
    plt.savefig(path + name + '.png')
    plt.close('all')
#======================

def subtract_LSfreq(t,f,ferr, n_peaks=1, min_per = 0.1, max_per = 500, samples_per_peak=50,add_model=False, plot=True):
    '''
    Subtract n max peaks from LS periodigram from the data
    '''
    from astropy.timeseries import LombScargle
    from scipy.optimize import curve_fit

    def phase_it(t, per, t0):
        phase = (t - t0)/per %1
        return phase

    t0=0.
    periods_composite = []
    #Model to be subtracted to search for other periods
    sin_model = lambda x,T,A: A*np.sin(2*np.pi*x/T)

    data_NGTS_tmasked = np.array([t,f,ferr])
    for i in range(n_peaks):
        frequency, power = LombScargle(*data_NGTS_tmasked).autopower(minimum_frequency=1/max_per,
                                                                     maximum_frequency=1/min_per,
                                                                     samples_per_peak=samples_per_peak)
        max_period, max_power = find_LSpeak(min_per, max_per, frequency, power)
        # print('Max Period: ', max_period)
        popt, pcov = curve_fit(sin_model, xdata=data_NGTS_tmasked[0], ydata=data_NGTS_tmasked[1],
                            p0=[max_period, np.std(data_NGTS_tmasked[1])], sigma=data_NGTS_tmasked[2], absolute_sigma=True)
        # print('Sine Per, Amp', popt)

        t_model_sin = np.linspace(data_NGTS_tmasked[0].min(),data_NGTS_tmasked[0].max(),
                                  data_NGTS_tmasked[0].size*10)
        t_model_phased = phase_it(t_model_sin,max_period, t0)
        t_model_phased_idx = np.argsort(t_model_phased)

        if plot:

            fig, axs = plt.subplots(2,1, figsize=(10,7), gridspec_kw={'height_ratios': [3.5,2.5]})
            axs[0].scatter(phase_it(data_NGTS_tmasked[0],max_period, t0), data_NGTS_tmasked[1],
                        c=np.sort(data_NGTS_tmasked[0]), s=5., cmap='viridis',alpha=1)
            axs[0].errorbar(*bin_data(phase_it(data_NGTS_tmasked[0],max_period, t0),
                                    data_NGTS_tmasked[1],np.ones(data_NGTS_tmasked[1].size),step=0.01), fmt='k.')
            if add_model:
                axs[0].plot(t_model_phased, sin_model(t_model_sin,*popt),'C3-',  linewidth=3)

            axs[1].plot(1/frequency, power/power.max(), 'C0-', label=f'Max power = {power.max()}')
            axs[1].axvspan(max_period-0.01*max_period,max_period+max_period*0.01, alpha=0.5, color='green',
                        label=r'P='+f'{max_period:.4f} d')
            axs[1].set_xscale('log')

            axs[1].legend(loc='lower right', frameon=False)
            axs[1].annotate(r'$P=$'+f'{max_period:.2f} d', xy=(.84,.82),xycoords='axes fraction', fontsize=20)
            #New dataset (residuals)
        data_NGTS_tmasked[1] -= sin_model(data_NGTS_tmasked[0], *popt)
        print('Sine Per, Amp', popt, 'subtracted from data')
    return data_NGTS_tmasked, max_period
    #Grab periods for creating a sine composite model
    #periods_composite.append(popt)
    #periods_composite = np.array(periods_composite)
