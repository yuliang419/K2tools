import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import floor, ceil
import warnings
from lmfit import minimize, Parameters, report_errors
from PyPDF2 import PdfFileReader, PdfFileWriter
import glob, os
import urllib
import matplotlib.image as mpimg
import model_transits


def sechmod(t, b, t0, w):
    """Fits a sech model to a transit as a faster, simpler alternative to the Mandel-Agol model.

    INPUTS:
        t - nd array of light curve times (days)
        b - 2*transit depth, defined as negative
        t0 - mid-transit time
        w - width of transit (days)

    RETURNS:
        nd array of model fluxes
    """
    warnings.simplefilter('ignore', RuntimeWarning)
    return 1 + b / (np.exp(-(t - t0) ** 2. / w ** 2.) + np.exp((t - t0) ** 2. / w ** 2.))


def rect_sechmod(t, b, t0, w, a0, a1):
    """Fits a sech model with linear detrending of background.

    INPUTS: see sechmod
        a0, a1 - coefficients of linear detrending function. The background is modelled as a0 + a1*t

    RETURNS: see sechmod
    """
    warnings.simplefilter('ignore', RuntimeWarning)
    return (1 + b / (np.exp(-(t - t0) ** 2. / w ** 2.) + np.exp((t - t0) ** 2. / w ** 2.))) * (a0 + a1 * t)


def residual(params, t, data, period=1, sech=True):
    """Residual function for fitting for midtransit times.

    INPUTS:
        params - lmfit.Parameters() object containing parameters to be fitted.
        t - nd array of light curve times (days).
        data - nd array of normalized light curve fluxes. Median out-of-transit flux should be set to 1.
        period - period of transit (days).
        sech - boolean. If True, will use sech model. Otherwise will fit Mandel-Agol model instead.
        The params argument must match the format of the model chosen.

    RETURNS:
        res - residual of data - model, to be used in lmfit.
    """

    if sech:
        vals = params.valuesdict()
        tc = vals['tc']
        b = vals['b']
        w = vals['w']
        a0 = vals['a0']
        a1 = vals['a1']
        model = rect_sechmod(t, b, tc, w, a0, a1)
    else:
        vals = params.valuesdict()
        tc = vals['tc']
        b = vals['b']
        r_a = vals['Rs_a']
        Rp_Rs = vals['Rp_Rs']
        F = vals['F']
        gamma1 = vals['gamma1']
        gamma2 = vals['gamma2']
        a0 = vals['a0']
        a1 = vals['a1']
        model = model_transits.modeltransit([tc, b, r_a, Rp_Rs, F, gamma1, gamma2], model_transits.occultquad, period,
                                            t)
        model *= (a0 + a1 * t)
    return data - model


def read_lc(name, p, t0, path='.'):
    """Read in data from file. If flux is not normalized, normalize first by setting mean of out-of-transit portion to 1.
    Creates plot handle for full light curve plot.

    INPUTS:
        name - name of light curve file to be read. Assuming the target file is named "[name].txt".
        p - best guess of transit period from BLS (days).
        t0 - best guess of transit ephemeris from BLS (JD).
        path - path of directory containing transit file.

    RETURNS:
        t - nd array of light curve times, with 3-sigma upward outliers in flux removed.
        f - nd array of light curve fluxes, with 3-sigma upward outliers in flux removed.
        newphase - nd array of light curve phases.
        w - estimated transit duration, expressed as fraction of transit period.
        depth - estimated transit depth (defined as negative).
        fig - plot handle for full light curve plot.
    """

    target = path + '/' + name + '.txt'
    t, f = np.loadtxt(target, unpack=True, usecols=(0, 1))

    fig = plt.figure(figsize=(12, 4))
    plt.plot(t, f, lw=0, marker='.')
    plt.xlabel('Time (days)')
    plt.ylabel('Relative flux')
    plt.ylim(min(f), max(f))
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    # plt.savefig('outputs/' + name + '_full_lc.pdf', dpi=150, bbox_inches='tight')

    # sort by phase to plot folded light curve (may not be necessary for 1-pager)
    t -= t0
    ph = t / p - np.around(t / p)
    order = sorted(range(len(ph)), key=lambda k: ph[k])
    phase = ph[order]
    f_sorted = f[order]
    t_sorted = t[order]

    p0 = [min(f_sorted) - 1., 0., 0.005]

    # fit sech model to folded light curve to get params
    popt, pcov = curve_fit(sechmod, phase, f_sorted, p0=p0)
    fmod = sechmod(phase, *popt)
    newphase = popt[1]
    w = popt[2]
    depth = popt[0] / 2.

    res = f_sorted - fmod
    sigma = np.std(res)
    print sigma

    # clip upward outliers
    good = np.where(f_sorted < 1 + 3 * sigma)
    f_sorted = f_sorted[good]
    t_sorted = t_sorted[good]

    order = sorted(range(len(t_sorted)), key=lambda k: t_sorted[k])
    t = t_sorted[order] + t0
    f = f_sorted[order]

    return t, f, newphase, w, depth, fig


def plot_indiv_trans(name, t, f, p, t0, window, p0, plotbad=True, plots=True, sech=True):
    """ Plot individual transits with a choice of sech or Mandel-Agol fit.

    INPUTS:
        t - nd array of light curve times.
        f - nd array of normalized light curve fluxes.
        window - approximate length of transit window (days). Include at least half a transit's worth of out-of-transit
        light curve on either side of dip.
        p0 - best guess of fit parameters. If sech, p0 = [w0, depth]. w0 is fractional width of transit from read_lc.
        If Mandel-Agol, p0 = [b,Rs_a,Rp_Rs,gamma1,gamma2].
        plotbad - set to True if you want to plot incomplete or misshapen transits along with good ones.
        sech - set to True if you want a sech fit. Otherwise use Mandel-Agol model.

    RETURNS:
        dt_all - nd array showing the time (days) to the nearest transit for each point.
        epochs - nd array of epoch number of each point.
        midpts - nd array of midtransit times associated with all points.
        err - array of errors on each midtransit time.
    """

    end = int(floor((t[-1] - t0) / p) + 1)
    start = int(floor((t[0] - t0) / p))
    cnt = 0
    epochs = []
    dt_all = []
    midpts = []
    err = []
    valid_trans = []

    params = Parameters()
    if sech:
        w0 = p0[0]
        depth = p0[1]
        params.add('tc', value=0, vary=True, min=-0.1, max=0.1)
        params.add('b', value=depth * 2, vary=False)
        params.add('w', value=w0 * p, vary=False)
        params.add('a0', value=1)
        params.add('a1', value=0)
    else:
        depth = -p0[2] ** 2
        params.add('tc', value=0, vary=True, min=-0.1, max=0.1)
        params.add('b', value=p0[0], vary=False)
        params.add('Rs_a', value=p0[1], vary=False)
        params.add('Rp_Rs', value=p0[2], vary=False)
        params.add('F', value=1, vary=False)
        params.add('gamma1', value=p0[3], vary=False)  # should I let these float?
        params.add('gamma2', value=p0[4], vary=False)
        params.add('a0', value=1, vary=True)
        params.add('a1', value=0, vary=True)

    for i in range(start, end):
        print 'Transit number ' + str(cnt)
        midt = i * p + t0

        dt = t - midt
        oot = np.where((abs(dt) > window) & (abs(dt) < window + 0.2 * p))[0]
        if len(oot) <= 1:
            continue

        fn = f / np.median(f[oot])

        select = np.where(abs(dt) < (window + 0.1 * p))[0]  # select single transit
        good = np.where(abs(dt) <= p / 2)[0]  # all points belonging to current transit

        if plots:
            if cnt % 8 == 0:
                plt.close('all')
                fig, ax = plt.subplots(8, figsize=(6, 12), sharex=True)

            if plotbad or (len(select) > 5):
                ax[cnt % 8].plot(dt[select], fn[select], lw=0, marker='.')
                ax[cnt % 8].axvline(x=0, color='k', ls='--')
                ax[cnt % 8].set_xlabel('Time from midtransit (days)')
                ax[cnt % 8].set_ylabel('Relative flux')
                ax[cnt % 8].set_ylim(1 + depth - 0.0003, 1 + 0.0003)
                ax[cnt % 8].set_xlim(-0.3, 0.3)
                ax[cnt % 8].locator_params(axis='y', nbins=5)
                ax[cnt % 8].get_yaxis().get_major_formatter().set_useOffset(False)
                ax[cnt % 8].annotate(str(cnt), xy=(0.85, 0.1), xycoords='axes fraction', size=15)

        dt_all += list(dt[good])

        if len(select) > 5:
            # fit sech to each transit

            try:
                fit = minimize(residual, params, args=(dt[select], fn[select], p, sech))
                fiterr = np.sqrt(fit.covar[0][0])
                err.append(fiterr)

                midpts += len(good) * [fit.params['tc'].value + i * p + t0]
                epochs += len(good) * [i]

                if plots:
                    tc = fit.params['tc'].value
                    a0 = fit.params['a0'].value
                    a1 = fit.params['a1'].value
                    tarr = np.linspace(dt[select][0], dt[select][-1], 200)
                    if sech:
                        fmod = rect_sechmod(tarr, depth * 2, tc, w0 * p, a0, a1)
                    else:
                        fmod = model_transits.modeltransit([fit.params['tc'].value, fit.params['b'].value,
                                                            fit.params['Rs_a'].value, fit.params['Rp_Rs'].value, 1,
                                                            fit.params['gamma1'].value,
                                                            fit.params['gamma2'].value], model_transits.occultquad, p,
                                                           tarr)
                        fmod *= (fit.params['a0'].value + fit.params['a1'].value * tarr)
                    ax[cnt % 8].plot(tarr, fmod, color='r')

                valid_trans.append(i)
            except TypeError:
                midpts += len(good) * [np.nan]
                epochs += len(good) * [np.nan]
                err.append(np.nan)
                print 'Fit failed'
                pass
        else:
            midpts += len(good) * [np.nan]
            err.append(np.nan)
            epochs += len(good) * [np.nan]
            print 'Too few data points'

        if plots and ((cnt % 8 == 7) or (i == end - 1)):
            plt.savefig('outputs/' + name + 'alltrans' + str(ceil(cnt / 8. + 0.01)) + '.pdf', dpi=150,
                        bbox_inches='tight')
        if plotbad or (len(select) > 5):
            cnt += 1

    print 'total transits:', cnt
    epochs = np.array(epochs)
    print 'good transits:', np.unique(epochs[np.where(~np.isnan(epochs))[0]])

    return np.array(dt_all), epochs, np.array(midpts), np.array(err)


def make_folded_lc(dt, f, epochs, midpts, window, fig=None):
    """Returns dt, flux, epochs and midpoints belonging to data points within transit windows.
    Makes plot of folded light curve if fig parameter is not None.

    :param dt: nd array showing the time (days) to the nearest transit for each point. Output dt_all of plot_indiv_trans.
    :param f: nd array of normalized fluxes.
    :param epochs: nd array of epoch numbers associated with all points.
    :param midpts: nd array of midtransit times associated with all points.
    :param window: approximate length of transit window (days). Include at least half a transit's worth of out-of-transit
        light curve on either side of dip.
    :param fig: plot handle indicating desired plot dimensions. e.g. fig = plt.figure(figsize=(10,4)).
    No plots will be made if set to None.

    :return:
        dt_tra - array of selected dt that fall within transit windows.
        f_tra - array of selected fluxes that fall within transit windows.
        epochs_tra - array of selected epochs that fall within transit windows.
        midpts_tra - array of selected midtransit times that fall within transit windows.
        fig - plot handle of folded transit.
    """
    transwindow = np.where(abs(dt) < window * 2.2)  # transit window size is hard to determine
    dt_tra = dt[transwindow]
    f_tra = f[transwindow]
    epochs_tra = epochs[transwindow]
    midpts_tra = midpts[transwindow]

    oot = np.where(abs(dt_tra) > window)[0]
    error = np.std(f_tra[oot])
    if fig is not None:
        plt.close('all')
        # fig = plt.figure(figsize=(10, 4))
        plt.plot(dt_tra, f_tra, lw=0, marker='.', color='b')
        plt.axvline(x=0, ls='--', color='k')
        plt.xlabel('t-tc (days)')
        plt.ylabel('Relative flux')

        # plt.savefig('outputs/' + name + '_folded.pdf', dpi=150)

    order = sorted(range(len(dt_tra)), key=lambda k: dt_tra[k])
    dt_tra = dt_tra[order]
    f_tra = f_tra[order]
    epochs_tra = epochs_tra[order]
    midpts_tra = midpts_tra[order]

    if fig is not None:
        return dt_tra, f_tra, epochs_tra, midpts_tra, fig
    else:
        return dt_tra, f_tra, epochs_tra, midpts_tra


def get_fold_fit(dt_tra, f_tra, depth, period, window, fig=None):
    """Uses lmfit to get a good estimate of the Mandel-Agol parameters from the folded light curve. The curve fitting
    routine will then be rerun using these better parameters.

    :param dt_tra: array of selected dt that fall within transit windows.
    :param f_tra: array of selected fluxes that fall within transit windows.
    :param depth: estimate of transit depth obtained from sech fit. Defined as negative.
    :param period: estimate of transit period (days).
    :param window: approximate length of transit window (days). Include at least half a transit's worth of out-of-transit
        light curve on either side of dip.
    :param fig: plot handle indicating desired plot dimensions. e.g. fig = plt.figure(figsize=(10,4)).
    No plots will be made if set to None.

    :return:
        fit - best-fit Mandel-Agol parameters from lmfit.minimise(). Contains the following params:
            tc - midtransit time. Centred at 0.
            b - impact parameter.
            Rs_a - radius of star/semimajor axis.
            F - out-of-transit flux, fixed at 1.
            gamma1, gamma2 - quadratic limb darkening parameters from Mandel & Agol (2002)
        fig - plot handle of folded transit with best-fit model.
    """
    # plotting can always be skipped now unless you want to debug
    params = Parameters()
    params.add('tc', value=0, vary=False, min=-0.1, max=0.1)
    params.add('b', value=0.7, vary=True)
    params.add('Rs_a', value=0.1, vary=True, min=0., max=0.5)
    params.add('Rp_Rs', value=(-depth) ** 0.5, vary=True)
    params.add('F', value=1, vary=False)
    params.add('gamma1', value=0.3, vary=True, min=0, max=0.5)  # should I let these float?
    params.add('gamma2', value=0.3, vary=True, min=0, max=0.5)
    params.add('a0', value=1, vary=False)
    params.add('a1', value=0, vary=False)

    fit = minimize(residual, params, args=(dt_tra, f_tra, period, False))
    tarr = np.linspace(min(dt_tra), max(dt_tra), 100)
    fmod = model_transits.modeltransit([fit.params['tc'].value, fit.params['b'].value, fit.params['Rs_a'].value,
                                        fit.params['Rp_Rs'].value, 1, fit.params['gamma1'].value,
                                        fit.params['gamma2'].value], model_transits.occultquad, period, tarr)

    if fig is not None:
        plt.close('all')
        plt.plot(dt_tra, f_tra, lw=0, marker='.')
        plt.plot(tarr, fmod, color='r')
        plt.axvline(x=-window, color='k', ls='--')
        plt.axvline(x=window, color='k', ls='--')
        plt.xlabel('Time from midtransit (days)')
        plt.ylabel('Relative flux')
    if fig is not None:
        return fit, fig
    else:
        return fit


def get_oc(all_epochs, all_midpts, err, fig=None):
    """Calculates accurate values for ephemeris and period. Plots O-C diagram if desired.

    :param all_epochs: nd array of epoch numbers associated with all points. From plot_indiv_trans.
    :param all_midpts: nd array of midtransit times associated with all points. From plot_indiv_trans.
    :param err: array of errors on midtransit times. One value for each unique time. From plot_indiv_trans.
    :param fig: plot handle indicating desired plot dimensions. e.g. fig = plt.figure(figsize=(10,4)).
    No plots will be made if set to None.

    :return:
        p_fit - best-fit transit period (days).
        t0_fit - best-fit transit ephemeris.
        fig - plot handle for O-C plot.
    """
    try:
        epochs = np.unique(all_epochs[np.where(~np.isnan(all_epochs))[0]])
        midpts = np.unique(all_midpts[np.where(~np.isnan(all_midpts))[0]])
        err = np.unique(err[np.where(~np.isnan(err))[0]])
    except:
        print('Error: invalid epochs and/or ephemerides')
        raise

    if len(epochs) > 2:
        coeffs, cov = np.polyfit(epochs, midpts, 1, cov=True)
        p_fit = coeffs[0]
        p_err = np.sqrt(cov[0, 0])
        t0_fit = coeffs[1]
        t0_err = np.sqrt(cov[1, 1])
    else:
        p_fit = (midpts[1] - midpts[0]) / (epochs[1] - epochs[0])
        p_err = 0
        t0_fit = (midpts[1] * epochs[0] - midpts[0] * epochs[1]) / (epochs[0] - epochs[1])
        t0_err = 0

    print 'p=', p_fit, '+-', p_err
    print 't0=', t0_fit, '+-', t0_err

    if len(epochs) > 2:
        fit = np.polyval(coeffs, epochs)
        oc = (midpts - fit) * 24.
    else:
        oc = midpts * 0

    err = np.array(err) * 24.
    if fig is not None:
        plt.close('all')
        # fig = plt.figure(figsize=(9, 4))
        plt.errorbar(epochs, oc, yerr=err, fmt='o')
        plt.axhline(color='k', ls='--')
        plt.ylabel('O-C (hours)')
        plt.xlabel('Epochs')
        plt.xlim(-0.1, max(epochs) + 1)
        # plt.savefig('outputs/' + name + '_oc.pdf', dpi=150, bbox_inches='tight')

    if fig is not None:
        return p_fit, t0_fit, fig
    else:
        return p_fit, t0_fit


def odd_even(dt_tra, f_tra, epochs_tra, window, period, p0):
    """Plots odd vs. even transits and calculates difference in depth.

    :param dt_tra: see get_fold_fit.
    :param f_tra: see get_fold_fit.
    :param epochs_tra: see get_fold_fit.
    :param window: see get_fold_fit.
    :param period: see get_fold_fit.
    :param p0: good estimate of Mandel-Agol parameters from get_fold_fit. p0 = [b, Rs_a, Rp_Rs, gamma1, gamma2]

    :return:
        fig - plot handle for odd-even comparison plot.
    """
    #
    odd = np.where(epochs_tra % 2 != 0)[0]
    even = np.where(epochs_tra % 2 == 0)[0]

    params = Parameters()
    params.add('tc', value=0, vary=False)
    params.add('b', value=p0[0], vary=True)
    params.add('Rs_a', value=p0[1], vary=True, min=0., max=0.5)
    params.add('Rp_Rs', value=p0[2], vary=True)
    params.add('F', value=1, vary=False)
    params.add('gamma1', value=p0[3], vary=False)
    params.add('gamma2', value=p0[4], vary=False)
    params.add('a0', value=1, vary=False)
    params.add('a1', value=0, vary=False)

    fit_odd = minimize(residual, params, args=(dt_tra[odd], f_tra[odd], period, False))
    fit_even = minimize(residual, params, args=(dt_tra[even], f_tra[even], period, False))

    oot = np.where(abs(dt_tra) > window)[0]
    sigma = np.std(dt_tra[oot])

    tarr = np.linspace(min(dt_tra), max(dt_tra), 200)
    oddmod = model_transits.modeltransit([fit_odd.params['tc'].value, fit_odd.params['b'].value,
                                          fit_odd.params['Rs_a'].value, fit_odd.params['Rp_Rs'].value, 1,
                                          fit_odd.params['gamma1'].value,
                                          fit_odd.params['gamma2'].value], model_transits.occultquad, period, tarr)
    evenmod = model_transits.modeltransit([fit_even.params['tc'].value, fit_even.params['b'].value,
                                           fit_even.params['Rs_a'].value, fit_even.params['Rp_Rs'].value, 1,
                                           fit_even.params['gamma1'].value,
                                           fit_even.params['gamma2'].value], model_transits.occultquad, period, tarr)
    odd_depth = min(oddmod)
    even_depth = min(evenmod)
    diff = abs(odd_depth - even_depth) / sigma

    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(13, 5))
    plt.subplots_adjust(wspace=0, hspace=0)
    ax1.plot(dt_tra[odd] * 24., f_tra[odd], lw=0, marker='.')
    ax1.plot(tarr * 24., oddmod, color='r')
    ax1.axhline(y=odd_depth, color='k', ls='--')
    ax1.set_xlabel('Time from midtransit (hours)')
    ax1.set_ylabel('Relative flux')
    ax1.set_xlim(min(dt_tra) * 24, max(dt_tra) * 24)
    ax1.annotate('Odd', xy=(0.75, 0.15), xycoords='axes fraction', size=15)

    ax2.plot(dt_tra[even] * 24., f_tra[even], lw=0, marker='.')
    ax2.plot(tarr * 24., evenmod, color='r')
    ax2.axhline(y=even_depth, color='k', ls='--')
    ax2.set_xlabel('Time from midtransit (hours)')
    ax2.set_xlim(min(dt_tra) * 24, max(dt_tra) * 24)
    ax2.annotate('Even', xy=(0.75, 0.15), xycoords='axes fraction', size=15)
    ax2.annotate('Diff: %.3f sigma' % diff, xy=(0.62, 0.05), xycoords='axes fraction', size=15)
    # plt.savefig('outputs/' + name + '_oddeven.pdf', dpi=150, bbox_inches='tight')
    return fig


def occultation(dt, f, p):
    """Plots folded light curve between two transits to check for secondary eclipses.

    :param dt: nd array showing the time (days) to the nearest transit for each point. Output dt_all of plot_indiv_trans.
    :param f: nd array of light curve flux.
    :param p: best-fit period (days).

    :return:
        fig - plot handle of secondary eclipse plot.
    """
    phase = dt / p
    phase[np.where(phase < 0)] += 1
    occ = np.where((phase > 0.2) & (phase < 0.8))
    ph_occ = phase[occ]
    f_occ = f[occ]

    tbins = np.linspace(0.2, 0.8, 51)
    fbin = []
    stddev = []
    for i in range(0, 50):
        inds = np.where((ph_occ >= tbins[i]) & (ph_occ < tbins[i + 1]))[0]
        fbin.append(np.mean(f_occ[inds]))
        stddev.append(np.std(f_occ[inds]))

    tbins = tbins[0:-1] + 0.6 / 50.

    plt.close('all')
    fig = plt.figure(figsize=(9, 4))
    plt.plot(ph_occ, f_occ, lw=0, marker='.', color='0.75')
    plt.plot(tbins, fbin, lw=2, color='r')
    plt.xlabel('Phase')
    plt.ylabel('Relative flux')
    plt.title('Occultation')
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    # plt.savefig('outputs/' + name + '_occult.pdf', dpi=150, bbox_inches='tight')
    return fig


def allparams(name, dat):
    # lists all best-fit params and prints to PDF
    # heading = '#P (days)  t0 (-2454900)  b  Rs_a  Rp/Rs Rp  Best-fit T'
    plt.close('all')
    fig = plt.figure(figsize=[7, 7])
    plt.plot([0, 1], [0, 1], lw=0)
    plt.axis('off')
    plt.annotate('P = ' + str(dat[0]) + ' days', xy=(0.1, 0.9), xycoords='axes fraction', size=20)
    plt.annotate('t0 (-2454900) = ' + str(dat[1]), xy=(0.1, 0.8), xycoords='axes fraction', size=20)
    plt.annotate('b = ' + str(dat[2]), xy=(0.1, 0.7), xycoords='axes fraction', size=20)
    plt.annotate('Rs/a = ' + str(dat[3]), xy=(0.1, 0.6), xycoords='axes fraction', size=20)
    plt.annotate('Rp/Rs = ' + str(dat[4]), xy=(0.1, 0.5), xycoords='axes fraction', size=20)
    plt.annotate('Rp = ' + str(dat[5]), xy=(0.1, 0.4), xycoords='axes fraction', size=20)
    plt.annotate('T from SED = ' + str(dat[6]) + ' K', xy=(0.1, 0.3), xycoords='axes fraction', size=20)
    plt.savefig('outputs/' + name + '_params.pdf', dpi=150)
