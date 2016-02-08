from pixel2flux import *
from clean_lc import *
import matplotlib.pyplot as plt
from numpy import *
import os
plt.ioff()

epic = 211400941
tref,xref,yref = read_ref('ref_centroid5.dat')

t,f,k,ra,dec = read_pixel(epic,5,'l')

if k>=18.5:
	cutoff_limit=1.5
elif k<13:
	cutoff_limit=5
else:
	cutoff_limit = 5-(3.5/5.5)*(k-13)

if k<=10.6:
	saturated = True
else:
	saturated = False

labels = find_aper(t,f,cutoff_limit=cutoff_limit,saturated=saturated)
seg = draw_aper(f,labels,epic)
t,f = remove_nan(t,f)

t,ftot,xc,yc = get_cen(t,f,labels,epic)
plot_lc(t,ftot,xc,yc,epic)

t,f,ftot,xc,yc,firetimes = remove_thrust(t,f,ftot,xc,yc)
tsegs = [[2305,2314.88615191],[2314.88615191,2320.77057466],[2320.77057466,2326.14417049],[2326.14417049,2332.06940302],[2332.06940302,\
2339.91516613],[2340.13991493,2344.79837308],[2344.79837308,2355.56585999],[2355.56585999,2360.10168765],[2360.10168765,2366.86457093],\
[2366.86457093,2372.72847053],[2372.72847053,2382]]
# tsegs = [[2144.103,2151.8672],[2151.8672,2157.2613],[2157.2613,2165.6383],[2165.6383,2173.4841],[2173.4841,2180.3283],\
# [2180.3283,2187.2142],[2187.2142,2190.6058],[2190.6058,2197.716],\
# [2197.716,2201.884],[2201.884,2206.2973],[2206.2973,2215]]

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

t3,ftot3,segs3 = spline(t,ftot,tsegs,squiggles=False,plot=False)
t3,f_corr3,xc3,yc3 = fit_lc(t3,ftot3,xc,yc,tsegs,tref,xref,yref,plot=e)
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

if mad1 < 0.9*mad3:
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

