from pixel2flux import *
from clean_lc import *
import matplotlib.pyplot as plt
from numpy import *
import os
plt.ioff()

epic = 211978909
tref,xref,yref = read_ref('ref_centroid5.dat')

t,f,k,ra,dec = read_pixel(epic,5,'l')
if k>14:
	cutoff_limit=3.5
else:
	cutoff_limit=5
labels = find_aper(t,f,cutoff_limit=cutoff_limit)
seg = draw_aper(f,labels,epic)
t,f = remove_nan(t,f)

t,ftot,xc,yc = get_cen(t,f,labels,epic)
plot_lc(t,ftot,xc,yc,epic)

t,ftot,xc,yc,firetimes = remove_thrust(t,ftot,xc,yc)
# tsegs = [[2231.4154,2238.5667],[2238.5667,2243.4499],[2243.4499,2251.6839],[2251.6839,2260.8577],[2260.8577,2268.6831],[2268.6831,2275.0782],\
# [2275.0782,2282.6586],[2282.6586,2290.2591],[2290.2591,t[-1]]]

tsegs = [[2306,2313.3939],[2313.3939,2318.788],[2318.788,2322.7313],[2322.7313,2330.1072],[2330.1072,2339.138],[2339.138,2344.7975],[2344.7975,2350.171],[2350.171,2355.2303],\
[2355.3403,2361.4493],[2361.4493,2368.8047],[2368.8047,2376.1602],[2376.1602,2381.4316]]


t1,ftot1,segs1 = spline(t,ftot,tsegs,squiggles=True)
t1,f_corr1,xc1,yc1 = fit_lc(t1,ftot1,xc,yc,tsegs,tref,xref,yref)
plt.close('all')
plt.clf()
plt.figure(figsize=(8,4))
plt.plot(t1,f_corr1,lw=0,marker='.')
plt.title('Aggressive detrending')
plt.xlabel('Time')
plt.ylabel('Flux')
plt.savefig('outputs/'+str(epic)+'_cleanlc_squiggles.pdf',bbox_inches='tight')

t2,ftot2,segs2 = spline(t,ftot,tsegs,squiggles=False)
t2,f_corr2,xc2,yc2 = fit_lc(t2,ftot2,xc,yc,tsegs,tref,xref,yref)
plt.close('all')
plt.clf()
plt.figure(figsize=(8,4))
plt.plot(t2,f_corr2,lw=0,marker='.')
plt.title('Non-aggressive detrending')
plt.xlabel('Time')
plt.ylabel('Flux')
plt.savefig('outputs/'+str(epic)+'_cleanlc.pdf',bbox_inches='tight')

mad1 = mad(f_corr1)
mad2 = mad(f_corr2)

if mad1<mad2:
	t = t1
	f_corr = f_corr1
	xc = xc1
	yc = yc1
	segs = segs1
	squiggle_note = 'chosen aggressive'
else:
	t = t2
	f_corr = f_corr2
	xc = xc2
	yc = yc2
	segs = segs2
	squiggle_note = 'chosen non-aggressive'

print squiggle_note


plt.close('all')
plt.clf()
plt.plot(xc,yc,lw=0,marker='.')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
plt.savefig('outputs/'+str(epic)+'_trail.pdf')

plotwrite(epic,k,ra,dec,squiggle_note,t,f_corr,segs)
pdf(str(epic))

