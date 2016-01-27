from pixel2flux import *
from clean_lc import *
import matplotlib.pyplot as plt
from numpy import *
import os
plt.ioff()

targs = open('pixel_files/targs.dat','r')
failed = []
tsegs = [[2144.103,2151.8672],[2151.8672,2157.2613],[2157.2613,2165.6383],[2165.6383,2173.4841],[2173.4841,2180.3283],\
[2180.3283,2187.2142],[2187.2142,2190.6058],[2190.6058,2197.716],\
[2197.716,2201.884],[2201.884,2206.2973],[2206.2973,2215]]
out = open('precision.dat','w')

for line in targs.readlines():
	x = line.split('-')[0]
	epic = x.split('o')[-1]
	epic = int(epic)
	
	print 'Working on target '+str(epic)

	tref,xref,yref = read_ref('ref_centroid.dat')

	try:
		t,f,k,ra,dec = read_pixel(epic,3,'l')
	except IOError:
		print 'Bad FITS file'
		continue

	if k>=18.5:
		cutoff_limit=2
	elif k<13:
		cutoff_limit=5
	else:
		cutoff_limit = 5-(3/5.5)*(k-13)

	if k<=10.6:
		saturated = True
	else:
		saturated = False
	labels = find_aper(t,f,cutoff_limit=cutoff_limit,saturated=saturated)

	if type(labels)==int:
		print 'failed aperture'
		failed.append(epic)
		continue
	seg = draw_aper(f,labels,epic)
	t,f = remove_nan(t,f)

	t,ftot,xc,yc = get_cen(t,f,labels,epic)
	plot_lc(t,ftot,xc,yc,epic)

	t,ftot,xc,yc,firetimes = remove_thrust(t,ftot,xc,yc)

	raw = open('rawlc/rawlc_'+str(epic)+".txt",'w')
	for i in range(0,len(t)):
		print>>raw, t[i],ftot[i]
	raw.close()
	# tsegs = [[2231.4154,2238.5667],[2238.5667,2243.4499],[2243.4499,2251.6839],[2251.6839,2260.8577],[2260.8577,2268.6831],[2268.6831,2275.0782],\
	# [2275.0782,2282.6586],[2282.6586,2290.2591],[2290.2591,t[-1]]]

	t1,ftot1,segs1 = spline(t,ftot,tsegs,squiggles=True)
	t1,f_corr1,xc1,yc1 = fit_lc(t1,ftot1,xc,yc,tsegs,tref,xref,yref)
	plt.close('all')
	plt.clf()
	plt.figure(figsize=(8,3))
	plt.plot(t1,f_corr1,lw=0,marker='.')
	plt.title('Aggressive detrending')
	plt.xlabel('Time')
	plt.ylabel('Flux')
	plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
	plt.savefig('outputs/'+str(epic)+'_cleanlc_squiggles.pdf',bbox_inches='tight')


	t3,ftot3,segs3 = spline(t,ftot,tsegs)
	t3,f_corr3,xc3,yc3 = fit_lc(t3,ftot3,xc,yc,tsegs,tref,xref,yref)
	plt.close('all')
	plt.clf()
	plt.figure(figsize=(8,3))
	plt.plot(t3,f_corr3,lw=0,marker='.')
	plt.title('Non-aggressive detrending')
	plt.xlabel('Time')
	plt.ylabel('Flux')
	plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
	plt.savefig('outputs/'+str(epic)+'_cleanlc.pdf',bbox_inches='tight')

	mad1 = mad(f_corr1)
	mad3 = mad(f_corr3)

	if mad1<mad3:
		t = t1
		f_corr = f_corr1
		xc = xc1
		yc = yc1
		segs = segs1
		squiggle_note = 'chosen aggressive'
	else:
		t = t3
		f_corr = f_corr3
		xc = xc3
		yc = yc3
		segs = segs3
		squiggle_note = 'chosen non-aggressive'

	error = std(f_corr3)
	good = where(abs(f_corr3-1)<3*error)
	error = std(f_corr3[good])
	print squiggle_note


	# plt.close('all')
	# plt.clf()
	# plt.plot(xc,yc,lw=0,marker='.')
	# plt.xlabel('X (pixels)')
	# plt.ylabel('Y (pixels)')
	# plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
	# plt.savefig('outputs/'+str(epic)+'_trail.pdf')

	plotwrite(epic,k,ra,dec,squiggle_note,t,f_corr,segs)
	# pdf(str(epic))

	print>>out, k, error

out.close()
targs.close()

