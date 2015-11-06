from numpy import *
import matplotlib.pyplot as plt
from lctools import *
from mcmc3 import *

epic = '210508766'
print 'Working on candidate',epic
file = open('f4candidates.dat','r')
for line in file.readlines():
	cols = line.split()
	if cols[0]==epic:
		initperiod = float(cols[1])
		depth = float(cols[2])/100.
		dur = initperiod*0.05
		break

t_guess=-0.25
badlist = [2,5,7,8,10,14,17,18,19,21,22,23]
t,f0,t0,newphase,w,depth = readlc(epic,initperiod,t_guess=t_guess)
excludewindow = 3*w*initperiod
print 'excludewindow=',excludewindow

t_good,dt,f,epochs,midpts,err = getlctrans(epic,t,f0,initperiod,t0,excludewindow,w,depth,badlist=badlist,plots=False)
dt_tra,f_tra,epochs_tra,midpts_tra = makefoldedlc(epic,dt,f,epochs,midpts,excludewindow,plots=False)
p0 = [depth*2,0,w*initperiod]
popt = getsechfold(dt_tra,f_tra,p0,plots=False)
period,t0 = getoc(epic,epochs,midpts,err,plots=False)

#repeat with better ephemeris
t_good,dt,f,epochs,midpts,err = getlctrans(epic,t,f0,period,t0,excludewindow,popt[2]/initperiod,popt[0]/2.,badlist=badlist)
dt_tra,f_tra,epochs_tra,midpts_tra = makefoldedlc(epic,dt,f,epochs,midpts,excludewindow,plots=False)
period,t0 = getoc(epic,epochs,midpts,err)
oddeven(epic,dt_tra,f_tra,epochs_tra)
occultation(epic,t,f0,period,t0,badlist)
# params = lmfit_trans(epic,dt_tra,f_tra,period,-popt[0]/2.,1)

# #now fit mcmc 
# Parameters are [T0,b,R_over_a,Rp_over_Rstar,flux_star,gamma1,gamma2]
transit_params = [0,0.5,0.5,-popt[0]/2,1,0.3,0.3]
oot = where(abs(dt_tra>excludewindow))[0]
error = std(f_tra[oot])
params = run_mcmc(epic,transit_params,period,-popt[0]/2,dt_tra,f_tra,error)
# params = median(params,axis=0)
Rs = 1. #(Rsun)
Rp = Rs*109.*params[3]


#write to file
heading = '#P (days)  t0 (-2454900)  b  R/a  Rp/Rs  gamma1  gamma2  Rp  Rp_err'
dat = [period, t0] + list(params) + [Rp]
savetxt('outputs/'+str(epic)+'fit.txt',dat,header=heading,fmt='%s')