import matplotlib.pyplot as plt
from numpy import *
from scipy import ndimage as ndi
from scipy import interpolate as itp
import warnings
from scipy.integrate import quad
from scipy.stats import binned_statistic
from PyPDF2 import PdfFileReader, PdfFileWriter
import glob, os
import urllib
import matplotlib.image as mpimg

def mad(data):
	up = where(data>median(data))[0]
	return median(abs(data[up] - median(data)))

def remove_thrust(time,flux,xc,yc,printtimes=False):
	#find and remove points in middle of thruster events, divide LC into segments
	diff_centroid = sqrt(diff(xc)**2 + diff(yc)**2)
	sigma = std(diff_centroid)

	thruster_mask = diff_centroid < 2*mean(diff_centroid)  #True=gap not in middle of thruster event
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

def spline(time,flux,tsegs,squiggles=False):
	#fit B-spline to light curve, remove outliers. Set squiggles=True for high stellar variability
	#takes really long for short cadence

	t_clean = array([])
	f_clean = array([])
	segs = []


	for i in range(0,len(tsegs)):
		fig = plt.figure()
		fig.clear()
		seg = tsegs[i]

		chunk = where( (time>=seg[0]) & (time<= seg[1]) )[0]
		
		tchunk = time[chunk]
		fchunk = flux[chunk]

		if squiggles:
			ind_knots = (linspace(3,len(tchunk)-3,15)).astype('int')
			knots = tchunk[ind_knots]
			# print 'knots', knots
			tck = itp.splrep(tchunk,fchunk,t=knots)	
		else:
			tck = itp.splrep(tchunk,fchunk,s=len(tchunk)) #+sqrt(2*len(tchunk)))
		fmod = itp.splev(tchunk,tck)


		#remove outliers, fit again
		j = 0
		while j<2:
			sig = mad(fchunk-fmod)
			good = where( abs(fmod-fchunk)<3*sig )[0]
			if squiggles:
				ind_knots = (linspace(3,len(tchunk)-3,15)).astype('int')
				knots = tchunk[ind_knots]
				try:
					tck = itp.splrep(tchunk[good],fchunk[good],t=knots)	
				except ValueError:
					tck = itp.splrep(tchunk[good],fchunk[good],s=len(tchunk[good])-3*sqrt(2*len(tchunk[good])))
			else:
				tck = itp.splrep(tchunk[good],fchunk[good],s=len(tchunk[good])+sqrt(2*len(tchunk[good])))
			fmod = itp.splev(tchunk,tck)
			
			j += 1

			# plt.close('all')
			# plt.plot(tchunk,fchunk,lw=0,marker='.')
			# plt.plot(tchunk,fmod)
			# plt.title('Spline fit '+str(seg[0]))
			# plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
			# plt.xlabel('Time')
			# plt.ylabel('Flux')
			# plt.show()

		fchunk -= array(fmod)
		fchunk += 1

		sig = std(fchunk-1)
		good = where( fchunk<(1+5*sig) )[0]
		fchunk = fchunk[good]
		tchunk = tchunk[good]

		t_clean = concatenate((t_clean,tchunk))
		f_clean = concatenate((f_clean,fchunk))
		segs += [i]*len(t_clean)


	# plt.close('all')
	# plt.plot(t_clean,stellar_act,lw=0,marker='.')
	# plt.xlabel('Time')
	# plt.ylabel('Flux')
	# plt.title('Stellar variability')
	# plt.show()

	return t_clean, f_clean, segs


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

def robust_fit(x,y,poly=True):
	order = [i[0] for i in sorted(enumerate(x), key=lambda k:k[1])]
	if poly:
		coeff = polyfit(x,y,4)
		yfit = polyval(coeff,x)
	else:
		ind_knots = (linspace(2,len(x)-2,13)).astype('int')
		knots = x[order][ind_knots]
		# print 'knots', knots

		tck = itp.splrep(x[order],y[order],t=knots)
		yfit= itp.splev(x,tck)	


	res = y-yfit
	sigma = std(res)
	good = where( abs(res)<3*sigma )[0]

	x_good = x[good]
	y_good = y[good]
	order = [i[0] for i in sorted(enumerate(x_good), key=lambda k:k[1])]

	if poly:
		coeffs = polyfit(x_good,y_good,4)
		yfit = polyval(coeffs,x)
	else:
		ind_knots = (linspace(3,len(good)-3,13)).astype('int')
		knots = x_good[order][ind_knots]
		# print 'knots', knots
		try:
			tck = itp.splrep(x_good[order],y_good[order],t=knots)
		except ValueError:
			tck = itp.splrep(x_good[order],y_good[order],s=len(x[good])-3*sqrt(2*len(x[good])))
		yfit= itp.splev(x,tck)

	return yfit

def robust_std(data):
	sig = std(data)
	good = where(abs(data)<3*sig)
	sig = std(data[good])
	return sig

def fit_lc(time,flux,xc,yc,tsegs,tref,xref,yref,plot=False):

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
				ind = where( abs(tref-t)==min(abs(tref-t)) )[0]
			xrefseg.append(xref[ind][0])
			yrefseg.append(yref[ind][0])

		xrefseg = array(xrefseg) 
		yrefseg = array(yrefseg) 
		plt.close('all')
		plt.plot(xrefseg,yrefseg,lw=0,marker='.')

		res = polyfit(xrefseg,yrefseg,1)
		yfit = polyval(res,xrefseg)


		# plt.show()
		s = (yrefseg-(yfit-res[-1])) / sqrt(1.+res[-2]**2.)
		h = (res[-2]*yrefseg+xrefseg) / sqrt(1.+res[-2]**2.)

		j = 0
		fchunk /= median(fchunk)


		while j<2:
			fit1 = robust_fit(h,fchunk)
			fit2 = robust_fit(h,fchunk,poly=False)
			sig1 = robust_std(fchunk-fit1)
			sig2 = robust_std(fchunk-fit2)

			if sig1>sig2*1.3: #prevent overfitting. don't use spline unless it's much better
				fit = fit2
			else:
				fit = fit1
			if plot:
				plt.close('all')
				plt.plot(h,fchunk,lw=0,marker='.')
				plt.title('Arclength '+str(tseg[0]))
				plt.plot(h,fit,color='r',lw=0,marker='o')
				plt.show()

			fchunk -= fit
			fchunk += 1

			fit1 = robust_fit(tchunk,fchunk)
			fit2 = robust_fit(tchunk,fchunk,poly=False)
			sig1 = robust_std(fchunk-fit1)
			sig2 = robust_std(fchunk-fit2)
			if sig1>sig2*1.3: #prevent overfitting. don't use spline unless it's much better
				fit = fit2
			else:
				fit = fit1

			if plot:
				plt.close('all')
				plt.plot(tchunk,fchunk,lw=0,marker='.')
				plt.title('Decorrelation against time '+str(tseg[0]))
				plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
				plt.plot(tchunk,fit,color='r',lw=0,marker='o')
				plt.show()
			fchunk -= fit
			fchunk += 1

			j += 1

		
		# fchunk += 1

		t_corr += list(tchunk)
		f_corr += list(fchunk)
		x_corr += list(xchunk)
		y_corr += list(ychunk)


	return array(t_corr),array(f_corr), array(x_corr), array(y_corr)

def plotwrite(epic,kepmag,ra,dec,squiggle_note,t,f_corr,segs):
	plt.close('all')
	fig = plt.figure(figsize=[8,8])
	plt.plot([0,1],[0,1],lw=0)
	plt.axis('off')
	plt.annotate('EPIC '+str(epic),xy=(0.1,0.9),xycoords='axes fraction',size=20)
	plt.annotate('Kepler mag = '+str(kepmag),xy=(0.1,0.8),xycoords='axes fraction',size=20)
	plt.annotate('ra = '+str(ra),xy=(0.1,0.7),xycoords='axes fraction',size=20)
	plt.annotate('dec = '+str(dec),xy=(0.1,0.6),xycoords='axes fraction',size=20)
	plt.annotate(squiggle_note,xy=(0.1,0.5),xycoords='axes fraction',size=20)
	plt.savefig('outputs/'+str(epic)+'_params.pdf',dpi=150)

	t -= 67
	file = open('DecorrelatedPhotometry/LCfluxesepic'+str(epic)+'star00.txt','w')
	for i in range(0,len(t)):
		print>>file, t[i], f_corr[i], segs[i]
	file.close()


def pdf(epic):
	os.chdir('outputs/')
	output = PdfFileWriter()
	page = output.addBlankPage(width=420,height=297)

	urllib.urlretrieve('https://cfop.ipac.caltech.edu/k2/files/'+epic+'/Finding_Chart/'+epic+'F-mc20150630.png',epic+'_chart.png')
	img = mpimg.imread(epic+'_chart.png')
	plt.close('all')
	plt.imshow(img)
	plt.axis('off')
	plt.savefig(epic+'_chart.pdf',dpi=250,bbox_inches='tight')

	obj = PdfFileReader(file(epic+'_chart.pdf','rb'))
	page1 = obj.getPage(0)
	page.mergeScaledTranslatedPage(page1,scale=0.8,tx=150,ty=-60)

	obj = PdfFileReader(file(epic+'_params.pdf','rb'))
	page1 = obj.getPage(0)
	page.mergeScaledTranslatedPage(page1,scale=0.18,tx=0,ty=180)

	obj = PdfFileReader(file(epic+'aper.pdf','rb'))
	page1 = obj.getPage(0)
	page.mergeScaledTranslatedPage(page1,scale=0.17,tx=70,ty=210)

	obj = PdfFileReader(file(epic+'_cleanlc_squiggles.pdf','rb'))
	page1 = obj.getPage(0)
	page.mergeScaledTranslatedPage(page1,scale=0.25,tx=10,ty=140)

	obj = PdfFileReader(file(epic+'_rawlc.pdf','rb'))
	page1 = obj.getPage(0)
	page.mergeScaledTranslatedPage(page1,scale=0.25,tx=155,ty=225)

	obj = PdfFileReader(file(epic+'_bg.pdf','rb'))
	page1 = obj.getPage(0)
	page.mergeScaledTranslatedPage(page1,scale=0.25,tx=290,ty=225)

	obj = PdfFileReader(file(epic+'_cleanlc.pdf','rb'))
	page1 = obj.getPage(0)
	page.mergeScaledTranslatedPage(page1,scale=0.25,tx=10,ty=75)

	# obj = PdfFileReader(file(epic+'_cleanlc.pdf','rb'))
	# page1 = obj.getPage(0)
	# page.mergeScaledTranslatedPage(page1,scale=0.25,tx=10,ty=10)

	trash = glob.glob(epic+'*.pdf')
	for f in trash:
		os.remove(f)
	os.remove(epic+'_chart.png')
		
	outstream = file(epic+'summary.pdf','wb')
	output.write(outstream)
	outstream.close()
	os.chdir('..')