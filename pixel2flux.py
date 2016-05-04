import matplotlib.pyplot as plt
from numpy import *
from scipy import ndimage as ndi
import pyfits
from skimage.morphology import watershed
from skimage.feature import peak_local_max as plm
from math import floor
from matplotlib.colors import LogNorm


def read_pixel(epic,field,cad):

	#cad = 'l' or 's'
	filename = 'pixel_files/ktwo'+str(epic)+'-c0'+str(field)+'_'+cad+'pd-targ.fits'
	#flux may contain nans

	hdulist = pyfits.open(filename)
	data = hdulist[1].data
	time = data['TIME']
	flux = data['FLUX']
	good = where(isnan(time)==0)
	time = time[good]
	flux = flux[good]
	header = hdulist[0].header

	kepmag = header['Kepmag']
	RA = header['RA_OBJ']
	DEC = header['DEC_OBJ']
	epic = header['KEPLERID']
	x = hdulist[2].header['CRVAL2P'] #x position of pixel
	y = hdulist[2].header['CRVAL1P'] #y position of pixel
	ra = hdulist[0].header['RA_OBJ']
	dec = hdulist[0].header['DEC_OBJ']

	if kepmag <= 10:
		print 'WARNING: saturated target'

	return time, flux, kepmag, ra, dec


def find_aper(time,flux,cutoff_limit=3,saturated=False):
	fsum = nansum(flux,axis=0) #sum over all images
	fsum -= min(fsum.ravel())
	cutoff = cutoff_limit*median(fsum)
	aper = array([fsum > cutoff])
	aper = 1*aper #arrays of 0s and 1s
	while sum(aper)<=1:
		print 'bad aper'
		cutoff_limit -= 0.2
		print cutoff_limit
		cutoff = cutoff_limit*median(fsum)
		aper = array([fsum > cutoff])
		aper = 1*aper #arrays of 0s and 1s
	size = sum(aper) #total no. of pixels in aperture

	if saturated:
		min_dist = aper.shape[1]/2
	else:
		min_dist = max([1,size**0.5/2.9])

	threshold_rel = 0.0002
	if cutoff_limit <2.5:
		threshold_rel /= 10 

	local_max = plm(fsum,indices=False,min_distance=min_dist,exclude_border=False,threshold_rel=threshold_rel) #threshold_rel determined by trial & error
	coords = plm(fsum,indices=True,min_distance=min_dist,exclude_border=False,threshold_rel=threshold_rel)

	dist = 100
	markers = ndi.label(local_max)[0]
	labels = watershed(-fsum,markers,mask=aper[0])

  
	for coord in coords:
		newdist = ((coord[0]-aper.shape[1]/2.)**2+(coord[1]-aper.shape[2]/2.)**2)**0.5
		if (newdist<dist) and (markers[coord[0],coord[1]] in unique(labels)):
			centnum = markers[coord[0],coord[1]]
			dist = newdist

	#in case there are more than one maxima
	if len(unique(labels))>2:
		labels = 1*(labels==centnum)

	labels /= labels.max()
	while sum(labels)<=1:
		cutoff_limit -= 0.2
		labels = find_aper(time,flux,cutoff_limit=cutoff_limit,saturated=False)

	return labels

def find_circ_aper(labels,rad,xc,yc): #use centroid position found from arbitrary aperture
	# first make a matrix that contains the x and y positions
	# remember x is second coordinate
	med_x = median(xc)
	med_y = median(yc)
  	x_pixels = [range(0,shape(labels)[1])] * shape(labels)[0]
  	y_pixels = transpose([range(0,shape(labels)[0])] * shape(labels)[1])

  	inside = ((x_pixels-med_x)**2. + (y_pixels-med_y)**2.)<rad**2. 
  	labels = 1*inside
  	return labels 


def draw_aper(flux,aper,epic):
	#input aperture from find_aper
	fsum = nansum(flux,axis=0)
	fsum -= min(fsum.ravel())
	plt.close('all')
	fig = plt.figure(figsize=(8,8))
	plt.imshow(fsum,norm=LogNorm(),interpolation='none')
	plt.set_cmap('gray')

	#find edge pixels in each row
	ver_seg = where(aper[:,1:] != aper[:,:-1])
	hor_seg = where(aper[1:,:] != aper[:-1,:])

	l = []
	for p in zip(*hor_seg):
		l.append((p[1],p[0]+1))
		l.append((p[1]+1,p[0]+1))
		l.append((nan,nan))

	for p in zip(*ver_seg):
		l.append((p[1]+1,p[0]))
		l.append((p[1]+1,p[0]+1))
		l.append((nan,nan))
	
	seg = array(l)
	x0 = -0.5
	x1 = aper.shape[1]+x0
	y0 = -0.5
	y1 = aper.shape[0]+y0

	if len(seg)!=0:
		seg[:,0] = x0 + (x1-x0)*seg[:,0]/aper.shape[1]
		seg[:,1] = y0 + (y1-y0)*seg[:,1]/aper.shape[0]
		plt.plot(seg[:,0],seg[:,1],color='r',zorder=10,lw=2.5)
		
	plt.xlim(0,aper.shape[1])
	plt.ylim(0,aper.shape[0])
	plt.savefig('outputs/'+str(epic)+'_aper.pdf',bbox_inches='tight')
	plt.close('all')

def remove_nan(time,flux):
#remove all empty frames
	bad = []
	for i in range(0,len(time)):
		f = array(flux[i])
		sig = nanstd(f)
		if isnan(sig)==1:
			bad.append(i)

	time = delete(time,bad)
	flux = delete(flux,bad,axis=0)
	return time, flux

def get_bg(time,flux,aper,epic,plot=True):
	inds = where(aper == 0)
	bg = []
	num = sum(aper)

	for i in range(0,len(time)):
		f = array(flux[i][inds])
		sig = nanstd(f)
		med = nanmedian(f)
		f[where(isnan(f)==1)] = 0
		f = f[where(abs(f-med) < 3*sig)] #clip 3sigma outliers
		#repeat
		sig = nanstd(f)
		med = nanmedian(f)
		f = f[where(abs(f-med) < 3*sig)] 

		med = nanmedian(f)
		bg.append(med)


	bg = array(bg)*num
	sig = nanstd(bg)
	med = nanmedian(bg)
	# bg[where(isnan(bg)==1)] = 0
	flagged = where(abs(bg-med)>7*sig)
	if plot:
		fig = plt.figure(figsize=(8,3))
		fig.clf()
		plt.plot(time,bg)
		plt.xlabel('Time')
		plt.ylabel('Background flux')
		plt.title('Background flux')
		plt.savefig('outputs/'+str(epic)+'_bg.pdf',bbox_inches='tight')

	return bg, flagged


def get_cen(time,flux,aper,epic):
	#find centroid and sum fluxes
	# bg, flags = get_bg(time,flux,aper,epic)
	# time = delete(time,flags)
	# flux = delete(flux,flags,axis=0)
	# bg = delete(bg,flags)

	xc = []
	yc = []
	ftot = []

	aperture_fluxes = flux*aper
  	
  	# sum over axis 2 and 1 (the X and Y positions), (axis 0 is the time)
  	f_t = nansum(nansum(aperture_fluxes,axis=2), axis=1) #- bg
  
 	# first make a matrix that contains the x and y positions
  	x_pixels = [range(0,shape(aperture_fluxes)[2])] * shape(aperture_fluxes)[1]
  	y_pixels = transpose([range(0,shape(aperture_fluxes)[1])] * shape(aperture_fluxes)[2])
  
  	# multiply the position matrix with the aperture fluxes to obtain x_i*f_i and y_i*f_i
  	xpos_times_flux = nansum( nansum( x_pixels*aperture_fluxes, axis=2), axis=1)
  	ypos_times_flux = nansum( nansum( y_pixels*aperture_fluxes, axis=2), axis=1)
  
  	# calculate centroids
  	xc = xpos_times_flux / f_t
  	yc = ypos_times_flux / f_t

  	ftot = f_t
	ftot = array(ftot)
	ftot /= median(ftot)

	return time, ftot, xc, yc


def plot_lc(time,ftot,xc,yc,epic):
	# plt.close('all')
	# fig = plt.figure(figsize=(8,3))
	# fig.clf()
	# plt.plot(time,ftot,marker='.',lw=0)
	# plt.ylim(max([min(ftot),0.9]),1.05)
	# plt.xlabel('Time')
	# plt.ylabel('Flux (pixel counts)')
	# plt.title('Raw light curve')
	# plt.savefig('outputs/'+str(epic)+'_rawlc.pdf',bbox_inches='tight')
	# plt.close(fig)

	fig = plt.figure()
	fig.clf()
	plt.plot(time,yc)
	plt.xlabel('Time')
	plt.ylabel('Horizontal centroid shift (pixels)')
	plt.savefig('outputs/'+str(epic)+'_ycentroid.pdf',bbox_inches='tight')
	plt.close(fig)

	fig = plt.figure()
	fig.clf()
	plt.plot(time,xc)
	plt.xlabel('Time')
	plt.ylabel('Vertical centroid shift (pixels)')
	plt.savefig('outputs/'+str(epic)+'_xcentroid.pdf',bbox_inches='tight')
	plt.close(fig)

	
