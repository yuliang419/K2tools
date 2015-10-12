import matplotlib.pyplot as plt
from numpy import *
from scipy import ndimage as ndi
from scipy import interpolate as itp
from math import sqrt


def remove_thrust(time,flux,xc,yc):

	#find and remove points in middle of thruster events, divide LC into segments
	diff_centroid = diff(xc)**2 + diff(yc)**2

	thruster_mask = diff_centroid < 1.5*mean(diff_centroid) #True=gap not in middle of thruster event
  	thruster_mask1 = insert(thruster_mask,0, False) #True=gap before is not thruster event
  	thruster_mask2 = append(thruster_mask,False) #True=gap after is not thruster event
  	thruster_mask = thruster_mask1*thruster_mask2 #True=gaps before and after are not thruster events
  	# thruster_mask *= 1

  	inds = where(thruster_mask == True)[0]
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


def spline(time,flux,xc,yc):
	#fit B-spline to light curve, remove outliers
	#doesn't work well for "extreme" squiggles

	time_chunks = arange(time[0],time[-1],5.)
	t_clean = array([])
	f_clean = array([])
	x_clean = array([])
	y_clean = array([])

	for i in range(0,len(time_chunks)-1):
		fig = plt.figure()
		fig.clear()

		chunk = where( (time>=time_chunks[i]) & (time<=time_chunks[i+1]) )[0]
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

		# tck = itp.splrep(tchunk,fchunk,s=len(tchunk)-sqrt(2*len(tchunk)),k=3)
		# fmod = itp.splev(tchunk,tck)
		# plt.plot(tchunk,fchunk,lw=0,marker='.')
		# plt.plot(tchunk,fmod,color='r')
		# plt.show()
	
		t_clean = concatenate((t_clean,tchunk))
		f_clean = concatenate((f_clean,fchunk))
		x_clean = concatenate((x_clean,xchunk))
		y_clean = concatenate((y_clean,ychunk))

	return t_clean, f_clean, x_clean, y_clean