from pixel2flux import *
from clean_lc import *
import matplotlib.pyplot as plt

epic = 210517600
squiggles = False

t,f,k,x,y = read_pixel(epic,4,'l')
labels = find_aper(t,f)
seg = draw_aper(f,labels,epic)
t,f = remove_nan(t,f)
bg,flags = get_bg(t,f,labels,epic)
t,ftot,xc,yc = plot_lc(t,f,labels,epic)
t,ftot,xc,yc = remove_thrust(t,ftot,xc,yc)
t,ftot,xc,yc = spline(t,ftot,xc,yc,squiggles=True)

plt.plot(xc,yc,lw=0,marker='.')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('outputs/'+str(epic)+'_trail.pdf')
plt.close('all')

plt.clf()
plt.plot(t,ftot,lw=0,marker='.')
plt.xlabel('Time')
plt.ylabel('Flux')
plt.savefig('outputs/'+str(epic)+'_cleanlc.pdf')
plt.close('all')

if squiggles:
	firetimes = [t[0],2232.685,2236.8099,2238.5671,2242.4492,2246.3722,2250.2962,2254.218,2258.3861,2260.8379,2264.9867,2269.9095, \
	2274.3228,2278.4908,2282.2911,2286.8473,2290.7498,2294.4275,2298.1256,t[-1]]
else:
	firetimes = [2229.,2233.6428,2238.567,2243.6749,2248.3743,2253.7275,2258.4064,2263.5347,2268.6835,2274.3226,2279.2466,2284.1502,2289.2785,2294.2025,2299.6782]

f_clean = fit_lc(t,ftot,xc,yc,firetimes)