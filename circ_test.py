from pixel2flux import *
from clean_lc import *
import matplotlib.pyplot as plt
from numpy import *
import os
plt.ioff()

targs = open('pixel_files/targs.dat','r')
failed = []
tref,xref,yref = read_ref('ref_centroid5.dat')
tsegs = [[2305,2314.88615191],[2314.88615191,2320.77057466],[2320.77057466,2326.14417049],[2326.14417049,2332.06940302],[2332.06940302,\
2339.91516613],[2340.13991493,2344.79837308],[2344.79837308,2355.56585999],[2355.56585999,2360.10168765],[2360.10168765,2366.86457093],\
[2366.86457093,2372.72847053],[2372.72847053,2382]]

for line in targs.readlines():
	plt.close('all')
	x = line.split('-')[0]
	epic = x.split('o')[-1]
	epic = int(epic)
	
	print 'Working on target '+str(epic)

	try:
		t,f,k,ra,dec = read_pixel(epic,5,'l')
	except IOError:
		print 'Bad FITS file'
		failed.append(epic)
		continue

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

		raw = open('rawlc/rawlc_'+str(epic)+".txt",'w')
		for i in range(0,len(t)):
			print>>raw, t[i],ftot[i]
		raw.close()

		t1,ftot1,segs1 = spline(t,ftot,tsegs,squiggles=True)
		tcorr1,f_corr1,xc1,yc1 = fit_lc(t1,ftot1,xc,yc,tsegs,tref,xref,yref,plot=False)

		t3,ftot3,segs3 = spline(t,ftot,tsegs,squiggles=False)
		tcorr3,f_corr3,xc3,yc3 = fit_lc(t3,ftot3,xc,yc,tsegs,tref,xref,yref,plot=False)

		mad1 = robust_std(f_corr1-1)
		mad3 = robust_std(f_corr3-1)

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

		if mad1<mad3:
			tcorr3 = tcorr1
			f_corr3 = f_corr1

		if rad<start_aper:
			dlim = min(best_f) - min_mad
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

	seg = draw_aper(f,best_labels,epic)
	best_t,best_f,best_segs = rem_min(best_t,best_f,best_segs)

	plotwrite(epic,k,ra,dec,best,best_t,best_f,best_segs)
	# pdf(str(epic))

