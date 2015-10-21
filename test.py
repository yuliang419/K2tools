from pixel2flux import *
from clean_lc import *
import matplotlib.pyplot as plt
from numpy import *

epic = 210303479
tsegs = [[t[0],2238.5667],[2242.4488,2271.8704],[2275.0578,t[-1]]]

t,f,k,x,y = read_pixel(epic,4,'l')
labels = find_aper(t,f)
seg = draw_aper(f,labels,epic)
t,f = remove_nan(t,f)

t,ftot,xc,yc = get_cen(t,f,labels,epic)
plot_lc(t,ftot,xc,yc)

t,ftot,xc,yc = remove_thrust(t,ftot,xc,yc)

t,ftot,xc,yc = spline(t,ftot,xc,yc,tsegs)

# plt.plot(xc,yc,lw=0,marker='.')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.savefig('outputs/'+str(epic)+'_trail.pdf')
# plt.close('all')

plt.clf()
plt.plot(t,ftot,lw=0,marker='.')
plt.xlabel('Time')
plt.ylabel('Flux')
plt.savefig('outputs/'+str(epic)+'_cleanlc.pdf')
plt.close('all')

if squiggles:
	firetimes = [2242.4492,2246.3722,2250.2962,2254.218,2258.3861,2260.8379,2264.9867,2269.9095, \
	2274.3228,2278.4908,2282.2911,2286.8473,2290.7498,2294.4275,2298.1256]

	#[2232.685,2236.8099,2238.5671,2242.4492,2246.3722,2250.2962,2254.218,2258.3861,2260.8379,2264.9867,2269.9095, \
	# 2274.3228,2278.4908,2282.2911,2286.8473,2290.7498,2294.4275,2298.1256]
else:
	firetimes = [2235.0935,2242.4899,2249.5594,2254.2587,2261.8185,2268.6835,2275.0786,2282.1886,2289.3193,2296.8995]

# t,f_corr,xc,yc = fit_lc(t,ftot,xc,yc,firetimes)