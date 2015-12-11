from numpy import *
import matplotlib.pyplot as plt
import pyfits
import urllib
from astropy.io import votable
from astropy.table import Table 
from astroquery.irsa_dust import IrsaDust
import astropy.coordinates as coord
import astropy.units as u
import os

# wav_ob = array([5.54e-1,6.48e-1,1.24,1.65, 2.19,3.35,4.60,1.16e1])*10000.0
# f_ob_orig = array([1.14e-3,2.25e-3,2.02e-2,2.50e-2,1.85e-2,9.98e-3,6.48e-3,1.06e-3])
# unc = array([1.0e-4, 5.09/104.0*2.25e-3,2.0e-3, 2.50e-3, 5.48/253.0*1.85e-2,1.97/89.3*9.98e-3,8.47/422.0*6.48e-3, 3.37/27.5*1.06e-3])
# reddening = array([0.325,0.257,0.086,0.054,0.037,0.023,0.018,0.0])*0.12/0.325 # in magnitude

def sed(epic='211800191',ra='132.884782',dec='17.319834'):
	os.chdir('outputs/')

	urllib.urlretrieve('http://vizier.u-strasbg.fr/viz-bin/sed?-c='+ra+"%2C"+dec+'&-c.rs=1', epic+'_sed.vot')
	print 'http://vizier.u-strasbg.fr/viz-bin/sed?-c='+ra+"%2C"+dec+'&-c.rs=0.005'

	tb = votable.parse_single_table(epic+'_sed.vot')
	data = tb.array
	wav_all = 3e5 * 1e4 / data['_sed_freq'].data #angstrom
	f_all = data['_sed_flux'].data
	unc_all = data['_sed_eflux'].data
	filters = data['_sed_filter'].data

	filter_dict = {'2MASS:Ks':'2MASS Ks','2MASS:J':'2MASS J','2MASS:H':'2MASS H','WISE:W1':'WISE-1','WISE:W2':'WISE-2','SDSS:u':'SDSS u','SDSS:g':'SDSS g',\
	'SDSS:r':'SDSS r','SDSS:i':'SDSS i','SDSS:z':'SDSS z'}

	c = coord.SkyCoord(float(ra)*u.deg,float(dec)*u.deg,frame='icrs')
	tb = IrsaDust.get_extinction_table(c)
	filters2 = tb['Filter_name']
	allA = tb['A_SandF']
	A = []
	f_ob_orig = []
	wav_ob = []
	unc = []

	for f in filters:
		if f in filter_dict.keys():
			filtmatch = filter_dict[f]
			ind = where(filters2==filtmatch)[0]
			A.append(mean(allA[ind]))
			ind = where(filters==f)[0]
			f_ob_orig.append(mean(f_all[ind]))
			wav_ob.append(mean(wav_all[ind]))
			unc.append(mean(unc_all[ind]))

	f_ob_orig = array(f_ob_orig)
	A = array(A)
	wav_ob = array(wav_ob)
	unc = array(unc)
	f_ob = (f_ob_orig*10**(A/2.5))

	metallicity = ['ckp00']
	m = [0.0]
	t = arange(3500,13000,250)
	t2 = arange(14000,50000,1000)
	t = concatenate((t,t2),axis=1)

	log_g = ['g20','g25','g30','g35','g40','g45','g50']
	g = arange(2.,5.,0.5)

	best_m =0
	best_t =0
	best_g =0
	best_off =0.0
	chi2_rec = 1e6
	os.chdir('..')

	for im, mval in enumerate(m):
		for it, tval in enumerate(t):
			for ig, gval in enumerate(g):
				#load model
				hdulist = pyfits.open('fits/'+metallicity[im]+'/'+metallicity[im]+'_'+str(tval)+'.fits')
				data = hdulist[1].data
				wmod = data['WAVELENGTH']
				fmod = data[log_g[ig]]*3.34e4*wmod**2

				#fit observations
				f_int = exp( interp(log(wav_ob), log(wmod), log(fmod)) )
				offsets = linspace(log(min(f_ob/f_int)), log(max(f_ob/f_int)), 51)
				for i_off, offset in enumerate(offsets):
					chi2 = sum((f_int*exp(offset)-f_ob)**2)

					print 'chi2=', chi2, mval, tval, gval
					if chi2 < chi2_rec:
						chi2_rec = chi2
						best_m = im
						best_g = ig
						best_t = it
						best_off = offset

	print 'best fit: m=', m[best_m], 'T=', t[best_t], 'log g=', g[best_g]

	hdulist = pyfits.open('fits/'+metallicity[best_m]+'/'+metallicity[best_m]+'_'+str(t[best_t])+'.fits')
	data = hdulist[1].data
	wmod = data['WAVELENGTH']
	fmod = data[log_g[best_g]]*3.34e4*wmod**2
	fmod *= exp(best_off)

	plt.close('all')
	fig = plt.figure(figsize=(8,5))
	plt.plot(wmod/1e4, fmod,label='Castelli & Kurucz model')
	plt.xscale('log')
	plt.plot(wav_ob/1e4, f_ob_orig, lw=0, marker='s', label='Uncorrected',ms=10)
	plt.plot(wav_ob/1e4, f_ob, lw=0, marker='o', label='Corrected for extinction',ms=10)
	plt.xlabel(r'${\rm Wavelength} \ (\mu m)}$',fontsize=18)
	plt.xlim(0.1,max(wmod)/1e4)
	plt.ylabel(r'$F_{\nu} \ {\rm (Jy)}$',fontsize=18)
	plt.legend()
	plt.savefig('outputs/'+epic+'_sed.pdf',dpi=150)
	return t[best_t]
