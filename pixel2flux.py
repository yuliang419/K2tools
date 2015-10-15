import matplotlib.pyplot as plt
from numpy import *
from scipy import ndimage as ndi
import pyfits
from skimage.morphology import watershed
from skimage.feature import peak_local_max as plm
from matplotlib.colors import LogNorm
from pylab import gray


def read_pixel(epic,field,cad):
	#cad = 'l' or 's'
	filename = 'ktwo'+str(epic)+'-c0'+str(field)+'_'+cad+'pd-targ.fits'
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

	heading = '#epic  kepmag  ra  dec'
	dat = [epic, kepmag, ra, dec]
	savetxt('outputs/'+str(epic)+'info.txt',dat,header=heading,fmt='%s')

	if kepmag <= 10:
		print 'WARNING: saturated target'

	return time, flux, kepmag, x, y


def find_aper(time,flux,cutoff_limit=3.5):
	fsum = nansum(flux,axis=0) #sum over all images
	cutoff = cutoff_limit*median(fsum)
	aper = array([fsum > cutoff])
	aper = 1*aper #arrays of 0s and 1s
	size = sum(aper) #total no. of pixels in aperture
	min_dist = max([1,size**0.5/2])

	local_max = plm(fsum,indices=False,min_distance=min_dist,exclude_border=False,threshold_rel=0.001) #threshold_rel determined by trial & error
	markers = ndi.label(local_max)[0]

	labels = watershed(fsum,markers,mask=aper[0])

	#in case there are more than one maxima
	if labels.max()>1:
		area = ndi.measurements.sum(aper,labels,index=arange(labels.max()+1))
		ind = where(area == max(area)) #pick out index with max area
		labels = 1*(labels==ind)

	return labels

def draw_aper(flux,aper,epic):
	#input aperture from find_aper
	fsum = nansum(flux,axis=0)
	plt.imshow(fsum,norm=LogNorm(),interpolation='none',cmap=gray())

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

	seg[:,0] = x0 + (x1-x0)*seg[:,0]/aper.shape[1]
	seg[:,1] = y0 + (y1-y0)*seg[:,1]/aper.shape[0]

	plt.plot(seg[:,0],seg[:,1],color='r',zorder=10,lw=2.5)
	plt.xlim(0,aper.shape[1])
	plt.ylim(0,aper.shape[0])
	plt.savefig('outputs/'+str(epic)+'aper.pdf')
	plt.close('all')

	return seg	

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
	flagged = where(abs(bg-med)>3*sig)
	if plot:
		fig = plt.figure()
		fig.clf()
		plt.plot(time,bg)
		plt.xlabel('Time')
		plt.ylabel('Background flux')
		plt.savefig('outputs/'+str(epic)+'_bg.pdf')
		plt.close(fig)

	return bg, flagged

def get_centroid(f,aper):
	#input flux of individual frames
	flux = f*aper
	f_tot = nansum(flux)

	pos = where(flux > 0) #coords of pixels in aperture
	x = pos[0]
	y = pos[1]

	xc = sum( flux[pos]*x )/f_tot
	yc = sum( flux[pos]*y )/f_tot

	# fig = plt.figure()
	# fig.clear()
	# plt.imshow(flux,norm=LogNorm(),interpolation='none')
	# plt.plot([yc],[xc],marker='o',color='b')
	# plt.show()	

	return xc,yc

def plot_lc(time,flux,aper,epic):
	bg, flags = get_bg(time,flux,aper,epic)
	time = delete(time,flags)
	flux = delete(flux,flags,axis=0)
	bg = delete(bg,flags)

	xc = []
	yc = []
	ftot = []

	for i in range(0,len(time)):
		f = array(flux[i])
		ftot.append(nansum(f*aper) - bg[i])

		x, y = get_centroid(f,aper)
		xc.append(x)
		yc.append(y)

	ftot = array(ftot)
	ftot /= median(ftot)
	xc = array(xc)
	yc = array(yc)
	fig = plt.figure()
	fig.clf()
	plt.plot(time,ftot,marker='.',lw=0)
	plt.xlabel('Time')
	plt.ylabel('Flux (pixel counts)')
	plt.savefig('outputs/'+str(epic)+'_rawlc.pdf')
	plt.close(fig)

	fig = plt.figure()
	fig.clf()
	plt.plot(time,yc)
	plt.xlabel('Time')
	plt.ylabel('Horizontal centroid shift (pixels)')
	plt.savefig('outputs/'+str(epic)+'_ycentroid.pdf')
	plt.close(fig)

	fig = plt.figure()
	fig.clf()
	plt.plot(time,xc)
	plt.xlabel('Time')
	plt.ylabel('Vertical centroid shift (pixels)')
	plt.savefig('outputs/'+str(epic)+'_xcentroid.pdf')
	plt.close(fig)

	return time, ftot, xc, yc