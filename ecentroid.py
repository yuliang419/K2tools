#Plots any selected star for the purpose of selecting good reference star for detrending
from pixel2flux import *
import matplotlib.pyplot as plt
from numpy import linspace, array
from clean_lc import remove_thrust

epic = 210303479
write = True
t,f,k,x,y = read_pixel(epic,4,'l')
labels = find_aper(t,f)
seg = draw_aper(f,labels,epic)
t,f = remove_nan(t,f)
t,ftot,xc,yc = get_cen(t,f,labels,epic)
print len(t)
t,ftot,xc,yc,firetimes = remove_thrust(t,ftot,xc,yc,printtimes=True)
print len(t)

fig, ax = plt.subplots(2,2)
ax[0,0].plot(t,ftot,lw=0,marker='.')
ax[0,0].set_xlabel('t')
ax[0,0].set_ylabel('Flux')

colors = ['r','y','g','c','b','m']
inds = linspace(0,len(t),7)
for i in range(len(inds)-1):
	start = inds[i]
	end = inds[i+1]
	ax[0,1].plot(xc[start:end],yc[start:end],marker='.',lw=0,color=colors[i])

ax[0,1].set_xlabel('x')
ax[0,1].set_ylabel('y')

ax[1,0].plot(t,xc,lw=0,marker='.')
ax[1,0].set_xlabel('t')
ax[1,0].set_ylabel('x')
for time in firetimes:
	plt.axvline(x=time,color='r')
ax[1,1].plot(t,yc,lw=0,marker='.')
ax[1,1].set_xlabel('t')
ax[1,1].set_ylabel('y')
plt.show()


if write:
	file = open('ref_centroid.dat','w')
	for i in range(0,len(xc)):
		print>>file, t[i], xc[i], yc[i]
	file.close()
