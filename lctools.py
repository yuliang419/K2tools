from numpy import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import floor
import warnings
from lmfit import minimize, Parameters, report_errors
from pytransit import MandelAgol

def sigma_clip(lim,f,*args):
	error = std(f)
	good = where( abs(f-median(f))<=lim*error )[0]
	for arg in args:
		arg = arg[good]

def sechmod(t,a,b,t0,w):
	warnings.simplefilter('ignore', RuntimeWarning)
	return a + b/(exp(-(t-t0)**2./w**2.) + exp((t-t0)**2./w**2.)) 

def readlc(epic,p,t_guess=0.):
	#read in data for given epic 
	target = 'DecorrelatedPhotometry/LCfluxesepic'+str(epic)+'star00.txt'
	t, f, seg = loadtxt(target,unpack=True) #dt=HJD-2454900
	good = where(seg>-10)[0]
	t = t[good]
	f = f[good]

	sigma_clip(10,f,t)

	plt.close('all')
	fig, ax = plt.subplots(2)
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

	p0 = [1.,min(f_sorted)-1,t_guess,0.005]
	popt, pcov = curve_fit(sechmod,phase,f_sorted,p0=p0)
	fmod = sechmod(phase,*popt)
	newphase = popt[2]
	w = popt[3]
	depth = popt[1]/2.

	ax[1].plot(phase,f_sorted,lw=0,marker='.')
	ax[1].plot(phase,fmod,color='r')
	ax[1].set_xlabel('Phase')
	ax[1].set_ylabel('Relative flux')
	ax[1].set_xlim(-0.5,0.5)
	ax[1].set_ylim(min(f),max(f))
	plt.show()
	t0 = t[0]-ph[0]*p+newphase*p

	return t,f,t0,newphase,w,depth

def getlctrans(t,f,p,t0,window,depth,badlist,plotbad=True,plots=True):
	end = int(floor(3000/p)+1)
	start = int(floor(-t0/p))
	cnt = 0
	epochs = []
	dt_good = []
	f_good = []
	midpts = []

	for i in range(start,start+end):
		midt = i*p + t0
		
		dt = t - midt
		oot = where( (abs(dt)>window) & (abs(dt)<window+0.1*p) )[0]
		if len(oot)<=1:
			continue

		fn = f/median(f[oot])
		#rectify
		coeffs = polyfit(dt[oot],fn[oot],2)
		fit = polyval(coeffs,dt)
		fn /= fit
		error = std(fn[oot]-fit[oot])
		select = where( abs(dt)<(window+0.1*p) )[0]

		good = where( (fn>1+depth-5*error) & (fn<1+3*error) & (abs(dt)<=p/2) )[0]
		if plots:
			if cnt%8 == 0:
				plt.close('all')
				fig, ax = plt.subplots(8,figsize=(8,14))
			
			if plotbad or ((len(select)>5) and (i not in badlist)==True):
				ax[cnt%8].plot(t[select],fn[select],lw=0,marker='.')
				ax[cnt%8].axvline(x=midt,color='k',ls='--')
				ax[cnt%8].set_xlabel('HJD-2454900')
				ax[cnt%8].set_ylabel('Relative flux')
				ax[cnt%8].set_ylim(min(fn[select])-error,max(fn[select]+error))
				ax[cnt%8].annotate(str(cnt),xy=(0.85,0.1),xycoords='axes fraction', size=15)

				if (cnt%8 == 7) or (i==(start+end-1)):
					plt.show()
			cnt += 1


		if (len(select)>5) and (i not in badlist):
			 epochs += len(good)*[i]
			 midpts += len(good)*[midt]
			 dt_good += list(dt[good])
			 f_good += list(fn[good])

	print 'total transits:', cnt
	print 'good transits:', unique(epochs)

	return array(dt_good), array(f_good), array(epochs), array(midpts)

def makefoldedlc(dt,f,epochs,midpts,window):
	transwindow = where( abs(dt)<window*1.8 )
	dt_tra = dt[transwindow]
	f_tra = f[transwindow]
	epochs_tra = epochs[transwindow]
	midpts_tra = midpts[transwindow]

	oot = where( abs(dt_tra)>1.3*window/2. )[0]
	error = std(f_tra[oot])
	plt.close('all')
	fig, ax = plt.subplots(2)
	ax[0].plot(dt_tra,f_tra,lw=0,marker='.')
	ax[0].axvline(x=0,ls='--',color='k')
	ax[0].set_xlabel('t (days)')
	ax[0].set_ylabel('Relative flux')
	ax[0].set_title('Before clipping')

	j = 0
	while j<2:
		good = where( (abs(dt_tra)<=1.3*window/2.) | (abs(f_tra-1)<3*error) )[0]
		dt_tra = dt_tra[good]
		f_tra = f_tra[good]
		epochs_tra = epochs_tra[good]
		midpts_tra = midpts_tra[good]
		j += 1

	ax[1].plot(dt_tra,f_tra,lw=0,marker='.')
	ax[1].axvline(x=0,ls='--',color='k')
	ax[1].set_xlabel('t (days)')
	ax[1].set_ylabel('Relative flux')
	ax[1].set_title('After clipping')
	plt.show()

	order = sorted(range(len(dt_tra)), key=lambda k: dt_tra[k])
	dt_tra = dt_tra[order]
	f_tra = f_tra[order]
	epochs_tra = epochs_tra[order]
	midpts_tra = midpts_tra[order]
	return dt_tra,f_tra,epochs_tra,midpts_tra

def getsechfold(dt_tra,f_tra,p0):
	#p0=[a,b,t0,w]
	popt, pcov = curve_fit(sechmod,dt_tra,f_tra,p0=p0)
	fmod = sechmod(dt_tra,*popt)
	plt.close('all')
	plt.plot(dt_tra,f_tra,lw=0,marker='.')
	plt.plot(dt_tra,fmod,color='r')
	plt.xlabel('t (days)')
	plt.ylabel('Relative flux')
	plt.show()
	return popt

def residual(params,t,data):
	#residual function for fitting for midtransit times
	vals = params.valuesdict()
	a = vals['a']
	b = vals['b']
	tc = vals['tc']
	w = vals['w']
	model = sechmod(t,a,b,tc,w)
	return (data-model)

def getoc(popt,dt_tra,f_tra,epochs_tra,midpts_tra,t0,p,plots=True):
	warnings.simplefilter('ignore', RankWarning)
	oc = []
	err = []
	cnt = 0
	tcs = []

	epochsord = sorted(unique(epochs_tra))
	params = Parameters()
	params.add('a',value=popt[0],vary=False)
	params.add('b',value=popt[1],vary=False)
	params.add('tc',value=0,vary=True,min=-0.05,max=0.05)
	params.add('w',value=popt[3],vary=False)

	for e in epochsord:
		now = where( epochs_tra==e )[0]
		dtnow = dt_tra[now]
		fnow = f_tra[now]

		fit = minimize(residual,params,args=(dtnow,fnow))
		fmod = fnow - residual(params,dtnow,fnow)
		fiterr = sqrt(fit.covar[0][0])
		err.append(fiterr)
		oc.append(fit.params['tc'].value)
		tcs.append(fit.params['tc'].value + e*p + t0)

		if plots:
			if cnt%8==0:
				plt.close('all')
				fig, ax = plt.subplots(8,figsize=(8,14))

			ax[cnt%8].plot(dtnow,fnow,lw=0,marker='.')
			ax[cnt%8].plot(dtnow,fmod,color='r')
			ax[cnt%8].axvline(x=0,color='k',ls='--')
			ax[cnt%8].set_xlabel('t (days)')
			ax[cnt%8].set_ylabel('Relative flux')
			ax[cnt%8].annotate("%.4f" % oc[-1],xy=(0.85,0.1),xycoords='axes fraction', size=15)

			if (cnt%8 == 7) or (e==epochsord[-1]):
				plt.show()
			cnt += 1

	coeffs,cov = polyfit(epochsord,tcs,1,cov=True)
	p_fit = coeffs[0]
	p_err = sqrt(cov[0,0])
	t0_fit = coeffs[1]
	t0_err = sqrt(cov[1,1])
	print 'p=',p_fit,'+-',p_err
	print 't0=',t0_fit,'+-',t0_err

	err = array(err)*24.
	oc = array(oc)*24.
	plt.close('all')
	fig = plt.figure(figsize=(10,10))
	plt.errorbar(epochsord,oc,yerr=err,fmt='o')
	plt.axhline(color='k',ls='--')
	plt.ylabel('O-C (hours)')
	plt.xlabel('Epochs')
	plt.show()

	return p_fit,t0_fit

def oddeven(dt_tra,f_tra,epochs_tra):
	odd = where(epochs_tra%2!=0)[0]
	even = where(epochs_tra%2==0)[0]

	plt.close('all')
	fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(15,5))
	ax1.plot(dt_tra[odd],f_tra[odd],lw=0,marker='.')
	ax1.axvline(color='k',ls='--')
	ax1.set_xlabel('t (days)')
	ax1.set_ylabel('Relative flux')
	ax1.annotate('Odd',xy=(0.85,0.1),xycoords='axes fraction', size=15)

	ax2.plot(dt_tra[even],f_tra[even],lw=0,marker='.')
	ax2.axvline(color='k',ls='--')
	ax2.set_xlabel('t (days)')
	ax2.annotate('Even',xy=(0.85,0.1),xycoords='axes fraction', size=15)
	plt.show()

