#Plots any selected star for the purpose of selecting good reference star for detrending
from pixel2flux import *
import matplotlib.pyplot as plt
from numpy import linspace, array
from clean_lc import remove_thrust
from sys import exit

epic = 212269339
write = True
t,f,k,x,y = read_pixel(epic,6,'l')
if  k>15:
	print 'Reference star too faint'
	exit()

labels = find_aper(t,f)
seg = draw_aper(f,labels,epic)
t,f = remove_nan(t,f)
t,ftot,xc,yc = get_cen(t,f,labels,epic)
print len(t)
t,f,ftot,xc,yc,firetimes = remove_thrust(t,f,ftot,xc,yc,printtimes=True)
print len(t)

outlier = where( (t>2431.21214655)&(t<2432.4584799) )

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
ax[0,1].plot(xc[outlier],yc[outlier],lw=0,marker='o',color='r')

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
