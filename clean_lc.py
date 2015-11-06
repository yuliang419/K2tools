import matplotlib.pyplot as plt
from numpy import *
from scipy import ndimage as ndi
from scipy import interpolate as itp
import warnings
from scipy.integrate import quad
from scipy.stats import binned_statistic
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
def clean_spline(x,y,squiggles):
	ind_knots = (linspace(3,len(x)-3,25)).astype('int')
	knots = x[ind_knots]
	if squiggles:
		tck = itp.splrep(x,y,t=knots)
	else:
		tck = itp.splrep(x,y,s=len(x)+sqrt(2*len(x)))
	ymod = itp.splev(x,tck)
	# plt.close('all')
	# plt.plot(x,y,lw=0,marker='.')
	# plt.plot(x,ymod,color='r')
	# plt.show()
	sig = std(y-ymod)
	good = where( abs(y-ymod) < 3*sig )[0]
	return good

def spline(time,flux,xc,yc,tsegs,squiggles=False):
	#fit B-spline to light curve, remove outliers. Set squiggles=True for high stellar variability
	#takes really long for short cadence

	t_clean = array([])
	f_clean = array([])
	x_clean = array([])
	y_clean = array([])

	for i in range(0,len(tsegs)):
		fig = plt.figure()
		fig.clear()
		seg = tsegs[i]

		chunk = where( (time>=seg[0]) & (time<= seg[1]) )[0]
		
		tchunk = time[chunk]
		fchunk = flux[chunk]
		xchunk = xc[chunk]
		ychunk = yc[chunk]
	
		#2 iterations of sigma clipping
		j = 0
		while j < 2:
			good = clean_spline(tchunk,fchunk,squiggles)
			tchunk = tchunk[good]
			fchunk = fchunk[good]
			xchunk = xchunk[good]
			ychunk = ychunk[good]
			j += 1
		

		ind_knots = (linspace(3,len(tchunk)-3,25)).astype('int')
		knots = tchunk[ind_knots]
		if squiggles:
			tck = itp.splrep(tchunk,fchunk,t=knots)	
		else:
			tck = itp.splrep(tchunk,fchunk,s=len(tchunk)+sqrt(2*len(tchunk)))
		fmod = itp.splev(tchunk,tck)
		fchunk /= array(fmod)
		# fchunk /= median(fchunk)
	
		t_clean = concatenate((t_clean,tchunk))
		f_clean = concatenate((f_clean,fchunk))
		x_clean = concatenate((x_clean,xchunk))
		y_clean = concatenate((y_clean,ychunk))

	return t_clean, f_clean, x_clean, y_clean

def read_ref(filename='ref_centroid.dat'):
	#read in reference centroid position and times
	tref = []
	xref = []
	yref = []
	file = open(filename,'r')
	for line in file.readlines():
		col = line.split()
		xref.append(float(col[1]))
		yref.append(float(col[2]))
		tref.append(float(col[0]))

	tref = array(tref)
	xref = array(xref)
	yref = array(yref)
	return tref, xref, yref

def fit_lc(time,flux,xc,yc,tsegs,tref,xref,yref):

	f_corr = []
	t_corr = []
	x_corr = []
	y_corr = []

	#firetimes gives times of thruster fires, so we take chunks that include integer numbers of drift segments
	for tseg in tsegs:
		warnings.simplefilter('ignore', RankWarning)
		chunk = where( (time>=tseg[0]) & (time<=tseg[1]) )[0]

		tchunk = time[chunk]
		fchunk = flux[chunk]
		xchunk = xc[chunk]
		ychunk = yc[chunk]
		
		xrefseg = []
		yrefseg = []
		for t in tchunk:
			ind = where( abs(tref-t)<0.01 )[0]
			if ind.size==0:
				ind = where( abs(tref-t)<0.03 )[0]
			xrefseg.append(xref[ind][0])
			yrefseg.append(yref[ind][0])

		xrefseg = array(xrefseg)
		yrefseg = array(yrefseg)
		# plt.close('all')
		# plt.plot(xrefseg,yrefseg,lw=0,marker='.')

		res = polyfit(xrefseg,yrefseg,4)
		yfit = polyval(res,xrefseg)

		# plt.show()
		s = (yrefseg-(yfit-res[-1])) / sqrt(1.+res[-2]**2.)
		h = (res[-2]*yrefseg+xrefseg) / sqrt(1.+res[-2]**2.)

		plt.close('all')
		plt.plot(tchunk,h,lw=0,marker='.')
		plt.xlabel('t')
		plt.ylabel('h')
		plt.show()

		j = 0

		while j<3:

			fit = polyfit(h,fchunk,4+j)
			mod = polyval(fit,h)
			sigma = std(fchunk-mod)

			plt.close('all')
			plt.plot(h,fchunk,lw=0,marker='.')
			plt.plot(h,mod,color='r',lw=0,marker='o')
			plt.annotate(str(tchunk[0])+'-'+str(tchunk[-1]),xy=(0.7,0.1),xycoords='axes fraction')
			# plt.plot(bin_edge[:-1],bin_med,color='r')
			plt.show()

			good = where( abs(fchunk-mod)<3*sigma )[0]
			h = h[good]
			fchunk = fchunk[good]
			tchunk = tchunk[good]
			xchunk = xchunk[good]
			ychunk = ychunk[good]
			mod = mod[good]
			fchunk /= mod

			fit = polyfit(tchunk,fchunk,5)
			mod = polyval(fit,tchunk)
			fchunk /= mod
			j += 1


		
		# fchunk /= median(fchunk)

		t_corr += list(tchunk)
		f_corr += list(fchunk)
		x_corr += list(xchunk)
		y_corr += list(ychunk)


	return array(t_corr),array(f_corr), array(x_corr), array(y_corr)