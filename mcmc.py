import emcee
import numpy as np
import matplotlib.pyplot as plt
import model_transits
from mcmc_priors import Prior_func, Flat_Prior_dis, Gauss_Prior_dis
import corner
from math import ceil
from lmfit import minimize, Parameters, report_errors


def logp(x, priors_info):
    return Prior_func(priors_info, x)


def transit_likelihoodfunc(params, period, times, dt_new, n_ev, data, sigma, priors_info):
    # PARAMS:
    # params[0, 1, 2, 3] = ['T0', 'b', 'R/a', 'Rp/Rs']  , for i = 0 ... len(a)-2 ---- the last ones are the
    # limb darkening, so len(params) = 1 + len(data)
    # params[len(params)-1] = ['q1','q2']

    u1_param = params[-2]
    u2_param = params[-1]

    # check physical boundaries for LD parameters and return -inf if they exceed those boundaries
    if ((u1_param + u2_param) > 1.) or (u1_param < 0.) or (u2_param < 0.):
        print 'unphysical limb darkening coeffs', u1_param, u2_param
        return -np.inf

    if params[3] < 0:
        print 'Rp/Rs<0'
        return -np.inf

    if params[2] < 0:
        print 'Rs/a<0'
        return -np.inf

    params[1] = abs(params[1])

    # calculate the likelihoods for all different transit models and sum them together
    log_likelihood = []
    prior = []

    prior.append(logp(np.array([params[0], params[1], params[2], params[3], u1_param, u2_param]),
                      priors_info))
    # the prior has to be evaluated for q1 and q2, NOT u1 and u2

    if not np.isfinite(prior[0]):
        return -np.inf

    new_params = np.array([params[0], params[1], params[2], params[3], 1, u1_param, u2_param])

    # oversample model
    model_oversamp = model_transits.modeltransit(new_params, model_transits.occultquad, period, dt_new)
    if type(model_oversamp) is int:
        return -np.inf

    transit_model = []
    for i in range(0, len(times)):
        mean_f = np.mean(model_oversamp[i * n_ev:i * n_ev + n_ev - 1])
        transit_model.append(mean_f)

    log_likelihood.append(
        -sum(((np.array(transit_model) - np.array(data)) ** 2. / ((2. ** 0.5) * np.array(sigma) ** 2)) +
             (np.log(np.array(sigma)) * (2. * np.pi) ** 0.5)))

    # print 'prior / likelihood'
    # print prior
    # print log_likelihood
    return sum(np.array(prior)) + sum(np.array(log_likelihood))


def residual(params, t, data, period):
    # residual function for fitting for midtransit times
    vals = params.valuesdict()
    t0 = vals['t0']
    b = vals['b']
    r_a = vals['r_a']
    Rp_Rs = vals['Rp_Rs']
    F = vals['F']
    gamma1 = vals['gamma1']
    gamma2 = vals['gamma2']
    model = model_transits.modeltransit([t0, b, r_a, Rp_Rs, F, gamma1, gamma2], model_transits.occultquad, period, t)
    return data - model


def getparams(times, data, depth, period):
    # use this to get a good initial guess of the params
    params = Parameters()  # fitting parameters, set to vary=false to fix
    params.add('t0', value=0., vary=True, min=-0.05, max=0.05)
    params.add('b', value=0.7, vary=True, min=-1, max=1)
    params.add('r_a', value=0.2, vary=True, min=0, max=0.5)
    params.add('Rp_Rs', value=depth ** 0.5, vary=True, min=0, max=0.5)
    params.add('F', value=1, vary=False)
    params.add('gamma1', value=0.3, vary=True, min=0.1, max=0.5)
    params.add('gamma2', value=0.3, vary=True, min=0.1, max=0.5)

    fit = minimize(residual, params, args=(times, data, period))
    vals = fit.params.valuesdict()
    return [vals['t0'], vals['b'], vals['r_a'], vals['Rp_Rs'], vals['F'], vals['gamma1'], vals['gamma2']]


def run_mcmc(name, transit_params, period, times, data, sigma, nwalkers=50, nthread=1, burnintime=100):
    # Parameters are [T0,b,R_over_a,Rp_over_Rstar,gamma1,gamma2]
    # need to check if I chose the right prior for each parameter
    T0_prior = ['gauss', 0, 0.1]  # should be around 0.
    b_prior = ['flat', -1 - transit_params[3], 1 + transit_params[3]]
    R_over_a_prior = ['flat', 0.01, 0.4]
    Rp_over_Rstar_prior = ['flat', 0.0001, 0.3]
    gamma1_prior = ['flat', 0.1, 0.5]
    gamma2_prior = ['flat', 0.1, 0.5]

    priors_info = [T0_prior, b_prior, R_over_a_prior, Rp_over_Rstar_prior, gamma1_prior, gamma2_prior]
    ndim = len(priors_info)


    # oversample to 30-min cadence
    cad = 29.4 / 60. / 24.  # cadence in days
    n_pt = len(times)

    n_ev = 25
    n_tot = n_ev * n_pt
    dt_new = np.zeros(n_tot)

    for i, this_t in enumerate(times):
        for i_ev in range(0, n_ev - 1):
            dt_new[i * n_ev + i_ev] = this_t + 1.0 / n_ev * cad * (i_ev - ceil(n_ev / 2))

    # Initialize sampler:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, transit_likelihoodfunc,
                                    args=[period, times, dt_new, n_ev, data, sigma, priors_info], threads=nthread)
    starting_pos = [transit_params + 1e-6 * np.random.randn(ndim) for i in
                    range(nwalkers)]  # all start out in more or less the location of our best guess.
    pos, prob, state = sampler.run_mcmc(starting_pos, 1000)

    samples = sampler.chain[:, burnintime:, :].reshape((-1, ndim))

    fig = corner.corner(samples, labels=['t0', 'b', 'R/a', 'Rp/Rs', 'gamma1', 'gamma2'])
    fig.savefig('outputs/' + name + '_triangle.png')

    parnames = ['t0', 'b', 'R/a', 'Rp/Rs', 'gamma1', 'gamma2']
    for i in range(0, ndim):
        print parnames[i], np.percentile(np.array(samples[:, i]), [(100 - 68.3) / 2, 50, 50 + 68.3 / 2])

    params = np.median(samples, axis=0)
    timearr = np.linspace(times[0], times[-1], 100)
    model_oversamp = model_transits.modeltransit([params[0], params[1], params[2], params[3], 1, params[4], params[5]],
                                                 model_transits.occultquad, period, dt_new)
    model = []
    for i in range(0, len(times)):
        mean_f = np.mean(model_oversamp[i * n_ev:i * n_ev + n_ev - 1])
        model.append(mean_f)

    model2 = np.interp(timearr, times, model)

    dt = np.linspace(min(times), max(times), 100)
    model = model_transits.modeltransit([params[0], params[1], params[2], params[3], 1, params[4], params[5]],
                                        model_transits.occultquad, period, dt)

    plt.close('all')
    fig = plt.figure(figsize=(11, 5))
    plt.plot(times * 24, data, lw=0, marker='.')
    plt.plot(dt * 24., model, color='g', ls='--', lw=1.5)
    plt.plot(timearr * 24., model2, color='r', lw=1.5)

    plt.xlabel('Time from midtransit (hours)')
    plt.ylabel('Relative flux')
    plt.savefig('outputs/' + name + '_finalfit.pdf', dpi=150, bbox_inches='tight')

    return params
