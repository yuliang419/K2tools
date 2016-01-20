from numpy import *
import matplotlib.pyplot as plt
from lctools import *
from mcmc3 import *
from vizier import sed

epic = '211418729'
# name = '211331236b'
print 'Working on candidate',epic
file = open('f5candidates.dat','r')
for line in file.readlines():
	cols = line.split()
	if cols[0]==epic:
		t0 = float(cols[1])
		initperiod = float(cols[2])
		depth = float(cols[3])
		dur = initperiod*0.05
		ra = cols[4]
		dec = cols[5]
		break

badlist = []
t,f0,newphase,w,depth = readlc(epic,initperiod,t0)
excludewindow = 3*abs(w)*initperiod
print 'excludewindow=',excludewindow

t_good,dt,f,epochs,midpts,err = getlctrans(epic,t,f0,initperiod,t0,excludewindow,w,depth,badlist=badlist,plots=False)
dt_tra,f_tra,epochs_tra,midpts_tra = makefoldedlc(epic,dt,f,epochs,midpts,excludewindow,plots=False)
p0 = [depth*2,0,w*initperiod]
popt = getsechfold(dt_tra,f_tra,p0,plots=False)
period,t0 = getoc(epic,epochs,midpts,err,plots=False)

#repeat with better ephemeris
t_good,dt,f,epochs,midpts,err = getlctrans(epic,t,f0,period,t0,excludewindow,popt[2]/initperiod,popt[0]/2.,badlist=badlist)
# t_good,dt,f,epochs,midpts,err = getlctrans(epic,t,f0,period,t0,excludewindow,w,depth,badlist=badlist)
dt_tra,f_tra,epochs_tra,midpts_tra = makefoldedlc(epic,dt,f,epochs,midpts,excludewindow,plots=False)
period,t0 = getoc(epic,epochs,midpts,err)
oddeven(epic,dt_tra,f_tra,epochs_tra)
occultation(epic,t,f0,period,t0,badlist)

# #now fit mcmc 
# Parameters are [T0,b,R_over_a,Rp_over_Rstar,flux_star,gamma1,gamma2] deoth=sqrt(abs(popt[0]/2))
transit_params = [0,0.9,0.5,sqrt(abs(popt[0]/2)),1,0.45,0.3]

# transit_params = [0,0.5,0.5,-depth,1,0.3,0.3]
oot = where(abs(dt_tra>excludewindow))[0]
error = std(f_tra[oot])

params = run_mcmc(epic,transit_params,period,dt_tra,f_tra,error)
# params = run_mcmc(epic,transit_params,period,-depth,dt_tra,f_tra,error)
Rs = 0.71 #(Rsun)
Rp = Rs*109.*params[3]
T = sed(epic,ra,dec)

#write to file
dat = [period, t0] + list(params[1:4]) + [Rp] + [T]
allparams(epic,dat)

merge(epic)
