import matplotlib.pyplot as plt
from numpy import *
from scipy import ndimage as ndi
from scipy import interpolate as itp
from math import sqrt
import warnings
# from lmfit import minimize, Parameters, report_errors, fit_report


def remove_thrust(time,flux,xc,yc):

	#find and remove points in middle of thruster events, divide LC into segments
	diff_centroid = diff(xc)**2 + diff(yc)**2

	thruster_mask = diff_centroid < 1.5*mean(diff_centroid) #True=gap not in middle of thruster event
  	thruster_mask1 = insert(thruster_mask,0, False) #True=gap before is not thruster event
  	thruster_mask2 = append(thruster_mask,False) #True=gap after is not thruster event
  	thruster_mask = thruster_mask1*thruster_mask2 #True=gaps before and after are not thruster events
  	# thruster_mask *= 1

  	inds = where(thruster_mask == True)[0]
  	# print 'fire times', time[where(thruster_mask==False)[0]]
  	time = time[inds]
  	flux = flux[inds]
  	xc = xc[inds]
  	yc = yc[inds]

  	return time, flux, xc, yc

#choose from B-spline or median filter to remove outliers
def clean_spline(x,y):
	tck = itp.splrep(x,y,s=len(x)-sqrt(2*len(x)),k=3)
	ymod = itp.splev(x,tck)
	sig = std(y-ymod)
	good = where( abs(y-ymod) < 3*sig )[0]
	return good

def spline(time,flux,xc,yc,squiggles=False):
	#fit B-spline to light curve, remove outliers
	#doesn't work well for short cadence, need different time 

	if squiggles:
		time_chunks = arange(time[0],time[-1]+3,3.)
	else:
		time_chunks = arange(time[0],time[-1]+5,5.)
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
	
		t_clean = concatenate((t_clean,tchunk))
		f_clean = concatenate((f_clean,fchunk))
		x_clean = concatenate((x_clean,xchunk))
		y_clean = concatenate((y_clean,ychunk))

	return t_clean, f_clean, x_clean, y_clean

# def residual(params,time,flux,xc,yc):
# 	#residual function used for lmfit
# 	parvals = params.valuesdict()
# 	x1 = parvals['x1']
# 	x2 = parvals['x2']
# 	x3 = parvals['x3']
# 	y1 = parvals['y1']
# 	y2 = parvals['y2']
# 	y3 = parvals['y3']
# 	xy1 = parvals['xy1']
# 	xy2 = parvals['xy2']
# 	t0 = parvals['t0']
# 	t1 = parvals['t1']
# 	t2 = parvals['t2']
# 	t3 = parvals['t3']
# 	t4 = parvals['t4']
# 	tamp = parvals['tamp']
# 	toff = parvals['toff']
# 	time0 = time[0]

# 	model = (t0 + tamp*sin((time-time0)+toff) + t1*(time-time0) + t2*((time-time0)**2.) + t3*((time-time0)**3.) + t4*((time-time0)**4.) \
# 		+ x1*xc + x2*xc**2. + x3*xc**3. + y1*yc + y2*yc**2. + y3*yc**3. + xy1*xc*yc + xy2*xc**2.*yc**2.)

# 	return array(flux-model)

def fit_lc(time,flux,xc,yc,firetimes):

	# params = Parameters()
	# params.add('x1',value=0.,vary=True)
	# params.add('x2',value=0.,vary=True)
	# params.add('x3',value=0.,vary=True)
	# params.add('y1',value=0.,vary=True)
	# params.add('y2',value=0.,vary=True)
	# params.add('y3',value=0.,vary=True)
	# params.add('xy1',value=0.,vary=True)
	# params.add('xy2',value=0.,vary=False)
	# params.add('t0',value=1.,vary=True)
	# params.add('t1',value=0.,vary=True)
	# params.add('t2',value=0.,vary=True)
	# params.add('t3',value=0.,vary=True)
	# params.add('t4',value=0.,vary=False)
	# params.add('tamp',value=0.,vary=False)
	# params.add('toff',value=0.,vary=False)
	f_corr = []

	#firetimes gives times of thruster fires, so we take chunks that include integer numbers of drift segments
	for i in range(0,len(firetimes)-1):
		warnings.simplefilter('ignore', RankWarning)
		chunk = where( (time>=firetimes[i]) & (time<=firetimes[i+1]) )[0]

		tchunk = time[chunk]
		fchunk = flux[chunk]
		xchunk = xc[chunk]
		ychunk = yc[chunk]

		coeffs = polyfit(tchunk,fchunk,4)
		mod = polyval(coeffs,tchunk)

		# fit = minimize(residual, params, args=(tchunk,fchunk,xchunk,ychunk))
		# fit = minimize(residual, params, args=(tchunk,fchunk,xchunk,ychunk))

		# mod = fchunk - residual(params,tchunk,fchunk,xchunk,ychunk)
	
		f_corr += [list(array(fchunk)-mod)]
		# plt.close('all')
		# plt.plot(tchunk,fchunk,lw=0,marker='.')
		# plt.plot(tchunk,mod,color='r',lw=2)
		# plt.show()

	f_corr = array(f_corr)/median(f_corr)
	return f_corr