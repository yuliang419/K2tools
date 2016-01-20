from numpy import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import floor,ceil 
import warnings
from lmfit import minimize, Parameters, report_errors
from matplotlib.ticker import *
from PyPDF2 import PdfFileReader, PdfFileWriter
import glob, os
import urllib
import matplotlib.image as mpimg
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

def readlc(epic,p,t0):
	#read in data for given epic 
	target = 'DecorrelatedPhotometry/LCfluxesepic'+str(epic)+'star00.txt'
	t, f, seg = loadtxt(target,unpack=True) #dt=HJD-2454900
	good = where(seg!=0)[0]
	t = t[good]
	f = f[good]

	plt.close('all')
	fig, ax = plt.subplots(2,figsize=(8,7))
	ax[0].plot(t,f,lw=0,marker='.')
	ax[0].set_xlabel('HJD-2454900')
	ax[0].set_ylabel('Relative flux')
	ax[0].set_ylim(min(f),1.005)
	ax[0].get_yaxis().get_major_formatter().set_useOffset(False)

	#sort by phase
	t = array(t) - t0
	ph = t/p - around(t/p)
	order = sorted(range(len(ph)), key=lambda k: ph[k])
	phase = ph[order]
	f_sorted = f[order]
	t_sorted = t[order]

	p0 = [min(f_sorted)-1,0,0.005]
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
	ax[1].set_ylim(min(f),1.005)

	res = f_sorted - fmod
	sigma = std(res)

	# res, [f_sorted, t_sorted, phase] = sigma_clip(7,res,f_sorted,t_sorted,phase)
	good = where(res<3*sigma)
	f_sorted = f_sorted[good]
	t_sorted = t_sorted[good]
	phase = phase[good]
	ax[1].plot(phase,f_sorted,color='b',lw=0,marker='.')
	ax[1].get_yaxis().get_major_formatter().set_useOffset(False)
	plt.savefig('outputs/'+epic+'_1firstcut.pdf',dpi=150,bbox_inches='tight')

	order = sorted(range(len(t_sorted)), key=lambda k: t_sorted[k])
	t = t_sorted[order] + t0
	f = f_sorted[order]

	return t,f,newphase,w,depth

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
	params.add('tc',value=0,vary=True,min=-0.1,max=0.1)
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
				fig, ax = plt.subplots(8,figsize=(6,12),sharex=True)
			
			if plotbad or ((len(select)>5) and (cnt not in badlist)==True):
				ax[cnt%8].plot(dt[select],fn[select],lw=0,marker='.')
				ax[cnt%8].axvline(x=0,color='k',ls='--')
				ax[cnt%8].set_xlabel('t-t0 (days)')
				ax[cnt%8].set_ylabel('Relative flux')
				ax[cnt%8].set_ylim(1+depth-0.0003,1+0.0003)
				ax[cnt%8].set_xlim(-0.3,0.3)
				ax[cnt%8].locator_params(axis='y',nbins=5)
				ax[cnt%8].get_yaxis().get_major_formatter().set_useOffset(False)
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

		if plots and ((cnt%8 == 7) or (i==end-1)):
			plt.savefig('outputs/'+epic+'_2alltrans'+str(ceil(cnt/8. + 0.01))+'.pdf',dpi=150,bbox_inches='tight')
		if plotbad or ((len(select)>5) and (i not in badlist)==True):
			print cnt
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
		plt.savefig('outputs/'+epic+'_3folded.pdf',dpi=150)

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
	if len(epochs)>2:
		coeffs,cov = polyfit(epochs,midpts,1,cov=True)
		p_fit = coeffs[0]
		p_err = sqrt(cov[0,0])
		t0_fit = coeffs[1]
		t0_err = sqrt(cov[1,1])
	else:
		p_fit = (midpts[1]-midpts[0])/(epochs[1]-epochs[0])
		p_err = 0
		t0_fit = (midpts[1]*epochs[0]-midpts[0]*epochs[1])/(epochs[0]-epochs[1])
		t0_err = 0

	print 'p=',p_fit,'+-',p_err
	print 't0=',t0_fit,'+-',t0_err

	if len(epochs)>2:
		fit = polyval(coeffs,epochs)
		oc = (midpts - fit)*24.
	else:
		oc = midpts*0

	err = array(err)*24.
	if plots:
		plt.close('all')
		fig = plt.figure(figsize=(9,4))
		plt.errorbar(epochs,oc,yerr=err,fmt='o')
		plt.axhline(color='k',ls='--')
		plt.ylabel('O-C (hours)')
		plt.xlabel('Epochs')
		plt.xlim(0,max(epochs)+1)
		plt.savefig('outputs/'+epic+'_4oc.pdf',dpi=150,bbox_inches='tight')
	return p_fit,t0_fit

def oddeven(epic,dt_tra,f_tra,epochs_tra):
	odd = where(epochs_tra%2!=0)[0]
	even = where(epochs_tra%2==0)[0]

	plt.close('all')
	fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(13,5))
	ax1.plot(dt_tra[odd]*24.,f_tra[odd],lw=0,marker='.')
	ax1.axvline(color='k',ls='--')
	ax1.set_xlabel('t-t0 (hours)')
	ax1.set_ylabel('Relative flux')
	ax1.set_xlim(min(dt_tra)*24,max(dt_tra)*24)
	ax1.annotate('Odd',xy=(0.85,0.1),xycoords='axes fraction', size=15)

	ax2.plot(dt_tra[even]*24.,f_tra[even],lw=0,marker='.')
	ax2.axvline(color='k',ls='--')
	ax2.set_xlabel('t-t0 (hours)')
	ax2.set_xlim(min(dt_tra)*24,max(dt_tra)*24)
	ax2.annotate('Even',xy=(0.85,0.1),xycoords='axes fraction', size=15)
	plt.savefig('outputs/'+epic+'_5oddeven.pdf',dpi=150,bbox_inches='tight')

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
	focc,tocc = sigma_clip(7,focc,tocc)

	tbins = linspace(0.1,0.9,51)
	fbin = []
	stddev = []
	for i in range(0,50):
		inds = where( (tocc >= tbins[i]) & (tocc < tbins[i+1]) )[0]
		fbin.append(mean( focc[inds] ))
		stddev.append(std( focc[inds] ))

	tbins = tbins[0:-1]+0.8/50.

	plt.close('all')
	fig = plt.figure(figsize=(9,4))
	plt.plot(tocc,focc,lw=0,marker='.',color='0.75')
	plt.plot(tbins,fbin,lw=2,color='r')
	plt.xlabel('Phase')
	plt.ylabel('Relative flux')
	plt.title('Occultation')
	plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
	plt.savefig('outputs/'+epic+'_6occult.pdf',dpi=150,bbox_inches='tight')

def allparams(epic,dat):
	#heading = '#P (days)  t0 (-2454900)  b  R/a  Rp/Rs Rp  Best-fit T'
	plt.close('all')
	fig = plt.figure(figsize=[7,7])
	plt.plot([0,1],[0,1],lw=0)
	plt.axis('off')
	plt.annotate('P = '+str(dat[0])+' days',xy=(0.1,0.9),xycoords='axes fraction',size=20)
	plt.annotate('t0 (-2454900) = '+str(dat[1]),xy=(0.1,0.8),xycoords='axes fraction',size=20)
	plt.annotate('b = '+str(dat[2]),xy=(0.1,0.7),xycoords='axes fraction',size=20)
	plt.annotate('R/a = '+str(dat[3]),xy=(0.1,0.6),xycoords='axes fraction',size=20)
	plt.annotate('Rp/Rs = '+str(dat[4]),xy=(0.1,0.5),xycoords='axes fraction',size=20)
	plt.annotate('Rp = '+str(dat[5]),xy=(0.1,0.4),xycoords='axes fraction',size=20)
	plt.annotate('T from SED = '+str(dat[6])+' K',xy=(0.1,0.3),xycoords='axes fraction',size=20)
	plt.savefig('outputs/'+epic+'_fit.pdf',dpi=150)

def merge(epic):
	os.chdir('outputs/')
	output = PdfFileWriter()
	page0 = output.addBlankPage(width=420,height=297)

	# obj = PdfFileReader(file(epic+'_chart.pdf','rb'))
	# page1 = obj.getPage(0)
	# page0.mergeScaledTranslatedPage(page1,scale=0.8,tx=150,ty=-60)

	# obj = PdfFileReader(file(epic+'_params.pdf','rb'))
	# page1 = obj.getPage(0)
	# page0.mergeScaledTranslatedPage(page1,scale=0.18,tx=0,ty=180)

	# obj = PdfFileReader(file(epic+'aper.pdf','rb'))
	# page1 = obj.getPage(0)
	# page0.mergeScaledTranslatedPage(page1,scale=0.17,tx=70,ty=210)

	# obj = PdfFileReader(file(epic+'_cleanlc_squiggles.pdf','rb'))
	# page1 = obj.getPage(0)
	# page0.mergeScaledTranslatedPage(page1,scale=0.25,tx=10,ty=140)

	# obj = PdfFileReader(file(epic+'_rawlc.pdf','rb'))
	# page1 = obj.getPage(0)
	# page0.mergeScaledTranslatedPage(page1,scale=0.25,tx=155,ty=225)

	# obj = PdfFileReader(file(epic+'_bg.pdf','rb'))
	# page1 = obj.getPage(0)
	# page0.mergeScaledTranslatedPage(page1,scale=0.25,tx=290,ty=225)

	# obj = PdfFileReader(file(epic+'_cleanlc.pdf','rb'))
	# page1 = obj.getPage(0)
	# page0.mergeScaledTranslatedPage(page1,scale=0.25,tx=10,ty=75)

	urllib.urlretrieve('https://cfop.ipac.caltech.edu/k2/files/'+epic+'/Finding_Chart/'+epic+'F-mc20150630.png',epic+'_chart.png')
	img = mpimg.imread(epic+'_chart.png')
	plt.close('all')
	plt.imshow(img)
	plt.axis('off')
	plt.savefig(epic+'_chart.pdf',dpi=250,bbox_inches='tight')

	page = output.addBlankPage(width=420,height=297)
	obj = PdfFileReader(file(epic+'_fit.pdf','rb'))
	page1 = obj.getPage(0)
	page.mergeScaledTranslatedPage(page1,scale=0.35,tx=150,ty=70)

	obj = PdfFileReader(file(epic+'_sed.pdf','rb'))
	page1 = obj.getPage(0)
	page.mergeScaledTranslatedPage(page1,scale=0.3,tx=170,ty=30)

	obj = PdfFileReader(file(epic+'_1firstcut.pdf','rb'))
	page1 = obj.getPage(0)
	page.mergeScaledTranslatedPage(page1,scale=0.29,tx=0,ty=155)

	obj = PdfFileReader(file(epic+'_7final.pdf','rb'))
	page1 = obj.getPage(0)
	page.mergeScaledTranslatedPage(page1,scale=0.2,tx=0,ty=10)

	obj = PdfFileReader(file(epic+'_4oc.pdf','rb'))
	page1 = obj.getPage(0)
	page.mergeScaledTranslatedPage(page1,scale=0.22,tx=150,ty=225)

	obj = PdfFileReader(file(epic+'_6occult.pdf','rb'))
	page1 = obj.getPage(0)
	page.mergeScaledTranslatedPage(page1,scale=0.22,tx=275,ty=225)

	obj = PdfFileReader(file(epic+'_5oddeven.pdf','rb'))
	page1 = obj.getPage(0)
	page.mergeScaledTranslatedPage(page1,scale=0.2,tx=0,ty=80)

	
	i = 0
	for filen in glob.glob(epic+'*alltrans*.pdf'):
		if i%3==0:
			page2 = output.addBlankPage(width=420,height=297)
		obj = PdfFileReader(file(filen,'rb'))
	 	page1 = obj.getPage(0)
	 	page2.mergeScaledTranslatedPage(page1,scale=0.35,tx=(i%3)*140,ty=20)
	 	i += 1

	# page3 = output.addBlankPage(width=420,height=297)
	# obj = PdfFileReader(file(epic+'_fit.pdf','rb'))
	# page1 = obj.getPage(0)
	# page3.mergeScaledTranslatedPage(page1,scale=0.35,tx=160,ty=120)
	# obj = PdfFileReader(file(epic+'_sed.pdf','rb'))
	# page1 = obj.getPage(0)
	# page3.mergeScaledTranslatedPage(page1,scale=0.3,tx=0,ty=155)

	trash = glob.glob(epic+'*.pdf')
	for f in trash:
		os.remove(f)
	os.remove(epic+'_chart.png')
		
	outstream = file(epic+'summary.pdf','wb')
	output.write(outstream)
	outstream.close()





