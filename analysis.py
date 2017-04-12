"""
    Written by Liang Yu, Jan 2017
    Included in this project and licensed under MIT.

    Takes as input a file containing time and detrended flux, and estimates of the transit ephemeris (JD-2454900)
    and period in days.
    Produces 1-page summary plots.

    Notes
    -------
    Create folder called "outputs" before running pipeline
    
    Usage
    -------
    python analysis.py [target_name] t0 initperiod
"""

# !/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import lctools as lc
import sys
# from mcmc import run_mcmc

name = sys.argv[1]
t0 = float(sys.argv[2])
initperiod = float(sys.argv[3])

t, f0, newphase, w, depth, full_lc = lc.read_lc(name, initperiod, t0)  # depth is defined to be negative

excludewindow = 3 * abs(w) * initperiod  # mask the transit to calculate std of out-of-transit region
print 'Masked region size: %.3f days' % excludewindow

# skip this step for 1-pager
dt, epochs, midpts, err, indiv_trans = lc.plot_indiv_trans(name, t, f0, initperiod, t0, excludewindow, p0=[w, depth], plots=False)
dt_tra, f_tra, epochs_tra, midpts_tra = lc.make_folded_lc(dt, f0, epochs, midpts, excludewindow)

fit = lc.get_fold_fit(dt_tra, f_tra, depth, initperiod, excludewindow)
period, t0 = lc.get_oc(epochs, midpts, err)  # redefine period and t0

# # repeat with better ephemeris
# p0 = [b,Rs_a,Rp_Rs,gamma1,gamma2]
t, epochs, midpts, err = lc.plot_indiv_trans(t, f0, period, t0, excludewindow, [fit.params['b'].value,
                                                                                         fit.params['Rs_a'].value,
                                                                                         fit.params['Rp_Rs'].value,
                                                                                         fit.params['gamma1'].value,
                                                                                         fit.params['gamma2'].value],
                                             sech=False)
dt_tra, f_tra, epochs_tra, midpts_tra, folded_lc = lc.make_folded_lc(dt, f0, epochs, midpts, excludewindow,
                                                                     fig=plt.figure(figsize=(10,4)))
period, t0, oc = lc.get_oc(epochs, midpts, err, fig=plt.figure(figsize=(9,4)))
oddeven = lc.odd_even(dt_tra, f_tra, epochs_tra, excludewindow, period, [fit.params['b'].value,
                                                                        fit.params['Rs_a'].value,
                                                                        fit.params['Rp_Rs'].value,
                                                                        fit.params['gamma1'].value,
                                                                        fit.params['gamma2'].value])
occult = lc.occultation(dt, f0, period)

# #now fit mcmc
# Parameters are [T0,b,R_over_a,Rp_over_Rstar,gamma1,gamma2] depth=fit.params['Rp_Rs'].value**2
transit_params = [0, fit.params['b'].value, fit.params['Rs_a'].value, fit.params['Rp_Rs'].value,
                  fit.params['gamma1'].value, fit.params['gamma2'].value]

oot = np.where(abs(dt_tra > excludewindow))[0]
error = np.std(f_tra[oot])

# params = run_mcmc(name, transit_params, period, dt_tra, f_tra, error)
