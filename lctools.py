from numpy import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import floor,ceil 
import warnings
from lmfit import minimize, Parameters, report_errors
from matplotlib.ticker import *
plt.ioff()

def sigma_clip(lim,f,*args):
	error = std(f)
	good = where( abs(f-median(f))<=lim*error )[0]
	f2 = f[good]
	arg2 = []
	for arg in args:
		arg2.append(arg[good])
	if len(args)==1:
		arg2 = arg2[0]
	return f2, arg2

def sechmod(t,b,t0,w):
	warnings.simplefilter('ignore', RuntimeWarning)
	return 1 + b/(exp(-(t-t0)**2./w**2.) + exp((t-t0)**2./w**2.)) 

def readlc(epic,p,t_guess=0.):
	#read in data for given epic 
	target = 'DecorrelatedPhotometry/LCfluxesepic'+str(epic)+'star00.txt'
	t, f, seg = loadtxt(target,unpack=True) #dt=HJD-2454900
	good = where(seg>-10)[0]
	t = t[good]
	f = f[good]

	f, t = sigma_clip(10,f,t)

	plt.close('all')
	fig, ax = plt.subplots(2,figsize=(8,7))
	ax[0].plot(t,f,lw=0,marker='.')
	ax[0].set_xlabel('HJD-2454900')
	ax[0].set_ylabel('Relative flux')
	ax[0].set_ylim(0.99,1.005)

	#sort by phase
	t = array(t)
	ph = t/p - around(t/p)
	order = sorted(range(len(ph)), key=lambda k: ph[k])
	phase = ph[order]
	f_sorted = f[order]
	t_sorted = t[order]

	p0 = [min(f_sorted)-1,t_guess,0.005]
	popt, pcov = curve_fit(sechmod,phase,f_sorted,p0=p0)
	fmod = sechmod(phase,*popt)
	newphase = popt[1]
	w = popt[2]
	depth = popt[0]/2.

	ax[1].plot(phase,f_sorted,lw=0,marker='.',color='0.75')
	ax[1].plot(phase,fmod,color='r',zorder=10)
	ax[1].set_xlabel('Phase')
	ax[1].set_ylabel('Relative flux')
	ax[1].set_xlim(-0.5,0.5)
	ax[1].set_ylim(min(f),max(f))
	t0 = t[0]-ph[0]*p+newphase*p

	res = f_sorted - fmod
	res, [f_sorted, t_sorted, phase] = sigma_clip(3,res,f_sorted,t_sorted,phase)
	ax[1].plot(phase,f_sorted,color='b',lw=0,marker='.')
	ax[1].get_yaxis().get_major_formatter().set_useOffset(False)
	plt.savefig('outputs/'+epic+'firstcut.pdf',dpi=150)

	order = sorted(range(len(t_sorted)), key=lambda k: t_sorted[k])
	t = t_sorted[order]
	f = f_sorted[order]

	return t,f,t0,newphase,w,depth

def getlctrans(epic,t,f,p,t0,window,w0,depth,badlist,plotbad=True,plots=True):
	end = int(floor( (t[-1]-t0)/p)+1)
	start = int(floor(-t0/p))
	cnt = 0
	epochs = []
	t_good = []
	dt_good = []
	f_good = []
	midpts = []
	err = []
	oc = []

	params = Parameters()
	params.add('b',value=depth*2,vary=False)
	params.add('tc',value=0,vary=True,min=-0.05,max=0.05)
	params.add('w',value=w0*p,vary=False)

	for i in range(start,end):
		midt = i*p + t0
		
		dt = t - midt
		oot = where( (abs(dt)>window) & (abs(dt)<window+0.1*p) )[0]
		if len(oot)<=1:
			continue

		fn = f/median(f[oot])
		#rectify
		coeffs = polyfit(dt[oot],fn[oot],3)
		fit = polyval(coeffs,dt)
		fn /= fit
		error = std(fn[oot]-fit[oot])
		select = where( abs(dt)<(window+0.1*p) )[0]

		good = where( (fn>1+depth-5*error) & (fn<1+3*error) & (abs(dt)<=p/2) )[0]
		if plots:
			if cnt%8 == 0:
				plt.close('all')
				fig, ax = plt.subplots(8,figsize=(8,13),sharex=True)
			
			if plotbad or ((len(select)>5) and (cnt not in badlist)==True):
				ax[cnt%8].plot(dt[select],fn[select],lw=0,marker='.')
				ax[cnt%8].axvline(x=0,color='k',ls='--')
				ax[cnt%8].set_xlabel('t-t0 (days)')
				ax[cnt%8].set_ylabel('Relative flux')
				ax[cnt%8].set_ylim(1+depth-0.0003,1+0.0003)
				ax[cnt%8].locator_params(axis='y',nbins=5)
				ax[cnt%8].annotate(str(cnt),xy=(0.85,0.1),xycoords='axes fraction', size=15)
				

		if (len(select)>5) and (cnt not in badlist):
			fit = minimize(residual,params,args=(dt[select],fn[select]))
			fmod = fn[select] - residual(params,dt[select],fn[select])
			fiterr = sqrt(fit.covar[0][0])
			err.append(fiterr)
			oc.append(fit.params['tc'].value)
			epochs += len(good)*[i]
			midpts += len(good)*[fit.params['tc'].value + i*p + t0]
			t_good += list(t[good])
			dt_good += list(dt[good])
			f_good += list(fn[good])

			if plots:
				tc = fit.params['tc'].value
				tarr = arange(dt[select][0],dt[select][-1],0.01)
				fmod = sechmod(tarr,depth*2,tc,w0*p)
				ax[cnt%8].plot(tarr,fmod,color='r')

		if plots and ((cnt%8 == 7) or (cnt==end-1)):
			plt.savefig('outputs/'+epic+'alltrans'+str(ceil(cnt/8.))+'.pdf',dpi=150)
		if plotbad or ((len(select)>5) and (i not in badlist)==True):
			cnt += 1

	print 'total transits:', cnt
	print 'good transits:', unique(epochs)

	return array(t_good), array(dt_good), array(f_good), array(epochs), array(midpts), array(err)

def makefoldedlc(epic,dt,f,epochs,midpts,window,plots=True):
	transwindow = where( abs(dt)<window*2.2)
	dt_tra = dt[transwindow]
	f_tra = f[transwindow]
	epochs_tra = epochs[transwindow]
	midpts_tra = midpts[transwindow]

	oot = where( abs(dt_tra)>1.3*window/2. )[0]
	error = std(f_tra[oot])
	if plots:
		plt.close('all')
		fig = plt.figure(figsize=(10,4))
		plt.plot(dt_tra,f_tra,lw=0,marker='.',color='0.75')
		plt.axvline(x=0,ls='--',color='k')
		plt.xlabel('t-t0 (days)')
		plt.ylabel('Relative flux')

	j = 0
	while j<2:
		good = where( (abs(dt_tra)<=1.3*window/2.) | (abs(f_tra-1)<3*error) )[0]
		dt_tra = dt_tra[good]
		f_tra = f_tra[good]
		epochs_tra = epochs_tra[good]
		midpts_tra = midpts_tra[good]
		j += 1

	if plots:
		plt.plot(dt_tra,f_tra,lw=0,marker='.',color='b')
		plt.savefig('outputs/'+epic+'folded.pdf',dpi=150)

	order = sorted(range(len(dt_tra)), key=lambda k: dt_tra[k])
	dt_tra = dt_tra[order]
	f_tra = f_tra[order]
	epochs_tra = epochs_tra[order]
	midpts_tra = midpts_tra[order]
	return dt_tra,f_tra,epochs_tra,midpts_tra

def getsechfold(dt_tra,f_tra,p0,plots=True):
	#p0=[a,b,t0,w]
	popt, pcov = curve_fit(sechmod,dt_tra,f_tra,p0=p0)
	fmod = sechmod(dt_tra,*popt)
	if plots:
		plt.close('all')
		plt.plot(dt_tra,f_tra,lw=0,marker='.')
		plt.plot(dt_tra,fmod,color='r')
		plt.xlabel('t-t0 (days)')
		plt.ylabel('Relative flux')
		plt.show()
	return popt

def residual(params,t,data):
	#residual function for fitting for midtransit times
	vals = params.valuesdict()
	b = vals['b']
	tc = vals['tc']
	w = vals['w']
	model = sechmod(t,b,tc,w)
	return (data-model)

def getoc(epic,epochs,midpts,err,plots=True):
	epochs = unique(epochs)
	midpts = unique(midpts)
	coeffs,cov = polyfit(epochs,midpts,1,cov=True)
	p_fit = coeffs[0]
	p_err = sqrt(cov[0,0])
	t0_fit = coeffs[1]
	t0_err = sqrt(cov[1,1])
	print 'p=',p_fit,'+-',p_err
	print 't0=',t0_fit,'+-',t0_err
	fit = polyval(coeffs,epochs)
	oc = (midpts - fit)*24.

	err = array(err)*24.
	if plots:
		plt.close('all')
		fig = plt.figure(figsize=(8,8))
		plt.errorbar(epochs,oc,yerr=err,fmt='o')
		plt.axhline(color='k',ls='--')
		plt.ylabel('O-C (hours)')
		plt.xlabel('Epochs')
		plt.savefig('outputs/'+epic+'oc.pdf',dpi=150)
	return p_fit,t0_fit

def oddeven(epic,dt_tra,f_tra,epochs_tra):
	odd = where(epochs_tra%2!=0)[0]
	even = where(epochs_tra%2==0)[0]

	plt.close('all')
	fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(13,5))
	ax1.plot(dt_tra[odd],f_tra[odd],lw=0,marker='.')
	ax1.axvline(color='k',ls='--')
	ax1.set_xlabel('t-t0 (days)')
	ax1.set_ylabel('Relative flux')
	ax1.annotate('Odd',xy=(0.85,0.1),xycoords='axes fraction', size=15)

	ax2.plot(dt_tra[even],f_tra[even],lw=0,marker='.')
	ax2.axvline(color='k',ls='--')
	ax2.set_xlabel('t-t0 (days)')
	ax2.annotate('Even',xy=(0.85,0.1),xycoords='axes fraction', size=15)
	plt.savefig('outputs/'+epic+'oddeven.pdf',dpi=150)

def occultation(epic,t,f,p,t0,badlist):
	end = int(floor(3000/p)+1)
	start = int(floor(-t0/p))
	tocc = []
	focc = []

	for i in range(start,start+end):
		if i in badlist:
			continue
		midt = i*p + t0
		dt = t - midt
		phase = dt / p
		ooe = where( (abs(phase-0.5)>0.05) & (abs(phase-0.5)<0.4) )[0]
		if len(ooe)==0:
			continue
		fn = f/median(f[ooe])
		#rectify
		coeffs = polyfit(dt[ooe],fn[ooe],2)
		fit = polyval(coeffs,dt)
		fn /= fit
		
		good = where( abs(phase-0.5)<0.4 )[0]
		tocc += list(phase[good])
		focc += list(fn[good])

	focc = array(focc)
	tocc = array(tocc)
	focc,tocc = sigma_clip(3,focc,tocc)
	plt.close('all')
	fig = plt.figure(figsize=(9,4))
	plt.plot(tocc,focc,lw=0,marker='.',color='r')
	plt.xlabel('Phase')
	plt.ylabel('Relative flux')
	plt.title('Occultation')
	plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
	plt.savefig('outputs/'+epic+'occult.pdf',dpi=150)

def transresidual(params,t,data,p):
	#residual function for fitting Mandel & Agol model
	# Parameters are [T0,b,R_over_a,Rp_over_Rstar,flux_star,gamma1,gamma2]
	vals = params.valuesdict()
	b = vals['b']
	a_over_R = vals['a_over_R']
	k = vals['k']
	inc = vals['inc']
	gamma1 = vals['gamma1']
	gamma2 = vals['gamma2']

	bmparams = batman.TransitParams()
	bmparams.t0 = 0
	bmparams.per = 0
	bmparams.rp = k
	bmparams.a = a_over_R
	bmparams.inc = inc
	bmparams.ecc = 0
	bmparams.w = 90
	bmparams.limb_dark = "quadratic"
	bmparams.u = [gamma1,gamma2]

	m = batman.TransitModel(bmparams,t)
	model = m.light_curve(bmparams)
	return (data-model)

def lmfit_trans(epic,t,f,p,depth,Rs):
	params = Parameters()
	params.add('b',value=0.5,vary=True,min=0,max=1)
	params.add('a_over_R',value=1/(pi*0.1),vary=True,min=1)
	params.add('k',value=(depth)**0.5,vary=True,min=0,max=(depth)**0.5+0.05)
	params.add('inc',value=45,vary=True,min=0,max=180)
	params.add('gamma1',value=0.3,vary=True,min=0.1,max=0.5)
	params.add('gamma2',value=0.3,vary=True,min=0.1,max=0.5)
	fit = minimize(transresidual,params,args=(t,f,p))

	params.add('b',value=fit.params['b'].value,vary=True,min=0,max=1)
	params.add('a_over_R',value=fit.params['a_over_R'].value,vary=True,min=1)
	params.add('k',value=fit.params['k'].value,vary=True,min=0,max=fit.params['k'].value+0.05)
	params.add('inc',value=fit.params['inc'],vary=True,min=0,max=180)
	params.add('gamma1',value=0.3,vary=True,min=0.1,max=0.5)
	params.add('gamma2',value=0.3,vary=True,min=0.1,max=0.5)
	fit = minimize(transresidual,params,args=(t,f,p))
	fmod = f - transresidual(params,t,f,p)

	plt.close('all')
	plt.plot(t*24.,f,lw=0,marker='.')
	plt.plot(t*24.,fmod,color='r')
	plt.xlabel('Time from midtransit (hr)')
	plt.ylabel('Relative flux')
	plt.show()
	return fit.params
