from numpy import *
import matplotlib.pyplot as plt
from lctools import *

epic = '210400751'
print 'Working on candidate',epic
file = open('f4candidates.dat','r')
for line in file.readlines():
	cols = line.split()
	if cols[0]==epic:
		initperiod = float(cols[1])
		depth = float(cols[2])/100.
		dur = initperiod*0.05
		break

t_guess=0.44
badlist = [5,6,10,13,14,15,16,17,19,20,21,23,27,30]
t,f0,t0,newphase,w,depth = readlc(epic,initperiod,t_guess=t_guess)
excludewindow = 3*w*initperiod
print 'excludewindow=',excludewindow

dt,f,epochs,midpts = getlctrans(t,f0,initperiod,t0,excludewindow,depth,badlist=badlist,plotbad=False,plots=False)

dt_tra,f_tra,epochs_tra,midpts_tra = makefoldedlc(dt,f,epochs,midpts,excludewindow)
p0 = [1,depth*2,0,w]
popt = getsechfold(dt_tra,f_tra,p0)
period,t0 = getoc(popt,dt_tra,f_tra,epochs_tra,midpts_tra,t0,initperiod,plots=False)

#repeat with better ephemeris
dt,f,epochs,midpts = getlctrans(t,f0,period,t0,excludewindow,depth,badlist=badlist,plotbad=False,plots=False)
dt_tra,f_tra,epochs_tra,midpts_tra = makefoldedlc(dt,f,epochs,midpts,excludewindow)
popt = getsechfold(dt_tra,f_tra,p0)
period,t0 = getoc(popt,dt_tra,f_tra,epochs_tra,midpts_tra,t0,period,plots=False)
oddeven(dt_tra,f_tra,epochs_tra)