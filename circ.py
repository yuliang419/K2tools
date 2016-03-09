from pixel2flux import *
from clean_lc import *
import matplotlib.pyplot as plt
from numpy import *
import os
plt.ioff()

targs = open('pixel_files/targs2127.dat','r')
# targs = open('vetted_para.txt','r')
failed = []

out = open('precision.dat','w')
tref,xref,yref = read_ref('ref_centroid.dat')
tsegs = [[2385,2392.2280283],[2392.2280283,2397.62213344],[2397.62213344,2405.95841839],[2405.95841839,2410.86208722],\
[2410.86208722,2415.80659931],[2415.80659931,2420.66937321],[2420.66937321,2424.61270132],[2424.61270132,2427.53444166],[2427.53444166,2430.8],\
[2432.4584799,2437.42337252],[2437.42337252,2443.00122187],[2443.00122187,2449.86626153],[2449.86626153,2455.03547282],[2455.03547282,\
2460.91980797],[2460.91980797,2465]]

for line in targs.readlines():
	if line.startswith('#'):
		continue
	plt.close('all')
	x = line.split('-')[0]
	epic = x.split('o')[-1]

	# cols = line.split()
	# epic = cols[0]
	epic = int(epic)
	
	print 'Working on target '+str(epic)

	try:
		t,f,k,ra,dec = read_pixel(epic,6,'l')
	except IOError:
		print 'Bad FITS file'
		failed.append(epic)
		continue

	print 'mag=', k


	if k>=18:
		cutoff_limit=1.5
		start_aper =1
	elif k<13:
		cutoff_limit=5
		start_aper = 3
	else:
		cutoff_limit = 5-(3.5/5)*(k-13)
		start_aper = 2

	if k<=10.6:
		saturated = True
	else:
		saturated = False

	if k<=10.:
		saturated = True
		start_aper += 2
	else:
		saturated = False


	#loop through apertures

	labels = find_aper(t,f,cutoff_limit=cutoff_limit,saturated=saturated)

	if type(labels)==int:
		print 'failed aperture'
		failed.append(epic)
		continue
	t,f = remove_nan(t,f)

	t,ftot,xc0,yc0 = get_cen(t,f,labels,epic)
	# plot_lc(t,ftot,xc0,yc0,epic)
	t,f,ftot,xc0,yc0,firetimes = remove_thrust(t,f,ftot,xc0,yc0)
	t0 = t
	xc = xc0
	yc = yc0


	apers = arange(start_aper-1,start_aper+3)
	min_mad = 100

	# make some subplots for raw and clean light curves
	plt.close('all')
	f_raw,axr = plt.subplots(4,figsize=(6,8),sharex=True)
	f_clean,axc = plt.subplots(4,figsize=(6,8),sharex=True)

	for rad in apers:
		if rad<start_aper:
			rad_note = 'custom'
		else:
			rad_note = str(rad)

		if rad>=start_aper:
			#rad=start_aper-1 is arbitrary aperture. do this step for circular apertures only
			labels = find_circ_aper(labels,rad,xc0,yc0)
			t,ftot,xc,yc = get_cen(t0,f,labels,epic)

		axr[rad-start_aper+1].plot(t,ftot,marker='.',lw=0)
		axr[rad-start_aper+1].set_ylabel('Flux (pixel counts)')
		axr[rad-start_aper+1].set_title('Raw light curve aper = '+rad_note)
		axr[rad-start_aper+1].get_yaxis().get_major_formatter().set_useOffset(False)


		t1,ftot1,segs1 = spline(t,ftot,tsegs,squiggles=True)
		tcorr1,f_corr1,xc1,yc1,raw1 = fit_lc(t1,ftot1,ftot,xc,yc,tsegs,tref,xref,yref,plot=False)

		t3,ftot3,segs3 = spline(t,ftot,tsegs,squiggles=False)
		tcorr3,f_corr3,xc3,yc3,raw3 = fit_lc(t3,ftot3,ftot,xc,yc,tsegs,tref,xref,yref,plot=False)

		mad1 = robust_std(f_corr1-1,transits=True)
		mad3 = robust_std(f_corr3-1,transits=True)

		if mad1 < min_mad:
			min_mad = mad1
			best_t = tcorr1
			best_f = f_corr1
			best_xc = xc1
			best_yc = yc1
			best_segs = segs1
			squiggle_note = 'Aggressive'
			best = squiggle_note+' '+rad_note
			best_labels = labels
			best_raw = raw1

		if mad3 < min_mad:
			min_mad = mad3
			best_t = tcorr3
			best_f = f_corr3
			best_xc = xc3
			best_yc = yc3
			best_segs = segs3
			squiggle_note = 'Non-aggressive'
			best_labels = labels
			best = squiggle_note+' '+rad_note
			best_raw = raw3

		if mad1<mad3:
			tcorr3 = tcorr1
			f_corr3 = f_corr1

		if rad<start_aper:
			dlim = max([min(best_f) - min_mad,0.94])
			ulim = max(best_f) + min_mad
			
		axc[rad-start_aper+1].plot(tcorr3,f_corr3,lw=0,marker='.')
		axc[rad-start_aper+1].set_title(squiggle_note+' aper = '+rad_note)
		axc[rad-start_aper+1].set_ylim(dlim,ulim)
		axc[rad-start_aper+1].set_ylabel('Flux')
		axc[rad-start_aper+1].get_yaxis().get_major_formatter().set_useOffset(False)
		
		print 'best result so far: '+best

	f_raw.savefig('outputs/'+str(epic)+'_rawlc.pdf',bbox_inches='tight')
	f_clean.savefig('outputs/'+str(epic)+'_cleanlc.pdf',bbox_inches='tight')
	plt.close(f_raw)
	plt.close(f_clean)

	draw_aper(f,best_labels,epic)
	best_t,best_f,best_segs,best_raw = rem_min(best_t,best_f,best_segs,best_raw)

	plotwrite(epic,k,ra,dec,best,best_t,best_f,best_segs,best_raw)

	# plt.close('all')
	# plt.plot(best_t,best_raw,lw=0,marker='.')
	# plt.show()

	print>>out, k, min_mad

out.close()

	# pdf(str(epic))

