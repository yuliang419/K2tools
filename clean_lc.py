import matplotlib.pyplot as plt
from numpy import *
from scipy import ndimage as ndi
from scipy import interpolate as itp
import warnings
from scipy.integrate import quad
# from lmfit import minimize, Parameters, report_errors, fit_report


def remove_thrust(time,flux,xc,yc,printtimes=False):

	#find and remove points in middle of thruster events, divide LC into segments
	diff_centroid = sqrt(diff(xc)**2 + diff(yc)**2)
	sigma = std(diff_centroid)

	thruster_mask = diff_centroid < median(diff_centroid)+3*sigma #True=gap not in middle of thruster event
  	thruster_mask1 = insert(thruster_mask,0, False) #True=gap before is not thruster event
  	thruster_mask2 = append(thruster_mask,False) #True=gap after is not thruster event
  	thruster_mask = thruster_mask1*thruster_mask2 #True=gaps before and after are not thruster events

  	time_thruster = time[ thruster_mask ]
  	diff_centroid_thruster = diff_centroid[ thruster_mask[1:] ]
  	firetimes = time[where(thruster_mask==False)[0]]

  	if printtimes:
  		print 'fire times', time[where(thruster_mask==False)[0]]
  	xc = xc[thruster_mask]
  	yc = yc[thruster_mask]
  	time = time[thruster_mask]
  	flux = flux[thruster_mask]

  	return time, flux, xc, yc, firetimes

#choose from B-spline or median filter to remove outliers
def clean_spline(x,y):
	tck = itp.splrep(x,y,s=len(x)-sqrt(2*len(x)),k=3)
	ymod = itp.splev(x,tck)
	sig = std(y-ymod)
	good = where( abs(y-ymod) < 3*sig )[0]
	return good

def spline(time,flux,xc,yc,tsegs):
	#fit B-spline to light curve, remove outliers
	#doesn't work well for short cadence, need different time 

	t_clean = array([])
	f_clean = array([])
	x_clean = array([])
	y_clean = array([])

	for i in range(0,len(time_chunks)-1):
		fig = plt.figure()
		fig.clear()

		chunk = where( (time>=time_chunks[i]) & (time<= time_chunks[i+1]) )[0]
		
		tchunk = time[chunk]
		fchunk = flux[chunk]
		xchunk = xc[chunk]
		ychunk = yc[chunk]
	
		#2 iterations of sigma clipping
		j = 0
		while j < 2:
			good = clean_spline(tchunk,fchunk)
			tchunk = tchunk[good]
			fchunk = fchunk[good]
			xchunk = xchunk[good]
			ychunk = ychunk[good]
			j += 1

		plt.close('all')
		if squiggles:
			tck = itp.splrep(tchunk,fchunk,s=len(tchunk)-sqrt(3*len(tchunk)),k=4)
		else:
			tck = itp.splrep(tchunk,fchunk,s=len(tchunk)-sqrt(2*len(tchunk)),k=3)
		fmod = itp.splev(tchunk,tck)
		
		# plt.plot(tchunk,fchunk,lw=0,marker='.')
		# plt.plot(tchunk,fmod,color='r')
		# plt.show()

		fchunk /= fmod
	
		t_clean = concatenate((t_clean,tchunk))
		f_clean = concatenate((f_clean,fchunk))
		x_clean = concatenate((x_clean,xchunk))
		y_clean = concatenate((y_clean,ychunk))

	return t_clean, f_clean, x_clean, y_clean


def fit_lc(time,flux,xc,yc,firetimes,bins=20.):

	f_corr = []
	t_corr = []
	x_corr = []
	y_corr = []

	#firetimes gives times of thruster fires, so we take chunks that include integer numbers of drift segments
	for i in range(0,len(firetimes)-1):
		warnings.simplefilter('ignore', RankWarning)
		chunk = where( (time>=firetimes[i]) & (time<=firetimes[i+1]) )[0]

		tchunk = time[chunk]
		fchunk = flux[chunk]
		xchunk = xc[chunk]
		ychunk = yc[chunk]


		#fit for x as function of y, then decorrelate against h
		# res = polyfit(xchunk,ychunk,4)
		# yfit = polyval(res,xchunk)
		# s = (ychunk-(yfit-res[-1])) / sqrt(1.+res[-2]**2.)
		# h = (res[-2]*ychunk+xchunk) / sqrt(1.+res[-2]**2.)

		res = polyfit(ychunk,xchunk,4)
		yfit = polyval(res,ychunk)
		s = (xchunk-(yfit-res[-1])) / sqrt(1.+res[-2]**2.)
		h = (res[-2]*xchunk+ychunk) / sqrt(1.+res[-2]**2.)

		bins = arange(min(h),max(h),(max(h)-min(h))/bins)
		med = []

		for i in range(0,len(bins)-1):
			fslice = fchunk[bins[i]:bins[i+1]]
			med.append(median(fslice))
		
		h_binned = h[bins[0:-1]+binsize/2]

		# fit = polyfit(h,fchunk,4)
		# mod = polyval(fit,h)

		plt.close('all')
		plt.plot(h,fchunk,lw=0,marker='.')
		plt.plot(h_binned,med,color='r')
		plt.show()

		# fchunk -= mod

		t_corr += list(tchunk)
		f_corr += list(fchunk)
		x_corr += list(xchunk)
		y_corr += list(ychunk)


	return array(t_corr),array(f_corr), array(x_corr), array(y_corr)