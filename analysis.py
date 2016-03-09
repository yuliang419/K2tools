from numpy import *
import matplotlib.pyplot as plt
from lctools import *
from mcmc3 import *
from vizier import sed

epic = '212300977'
# name = '212684743b'
print 'Working on candidate',epic
file = open('vetted_para.txt','r')
for line in file.readlines():
	if not line.startswith('#'):
		cols = line.split()
		if cols[0]==epic:
			t0 = float(cols[3])
			initperiod = float(cols[2])
			depth = float(cols[5])
			dur = initperiod*0.05
			# ra = cols[4]
			# dec = cols[5]
			break

badlist = [0,8,9]
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
# Parameters are [T0,b,R_over_a,Rp_over_Rstar,flux_star,gamma1,gamma2] depth=sqrt(abs(popt[0]/2))
transit_params = getparams(dt_tra,f_tra,abs(popt[0]/2),period)

# transit_params = [0,0.5,0.5,-depth,1,0.3,0.3]
oot = where(abs(dt_tra>excludewindow))[0]
error = std(f_tra[oot])

params = run_mcmc(epic,transit_params,period,dt_tra,f_tra,error)
# params = run_mcmc(epic,transit_params,period,-depth,dt_tra,f_tra,error)
Rs = 1.085 #(Rsun)
Rp = Rs*109.*params[3]

ra = '203.758114'
dec = '-17.503464'
T = sed(epic,ra,dec)

# #write to file
dat = [period, t0] + list(params[1:4]) + [Rp] + [T]
allparams(epic,dat)

merge(epic)
