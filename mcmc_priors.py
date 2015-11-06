from numpy import *
from scipy.special import erfcinv

def Flat_Prior_dis(r,x1,x2):
	return x1+r*(x2-x1)

def Jeff_Prior_dis(r,xmin,xmax):
	if r <= 0:
		return -1e32
	else:
		if xmin==0:
			xmin = 1e-10
		return exp(r*log(xmax/xmin) + log(xmin))

def Gauss_Prior_dis(r,mu,sigma):
    if (r <= 1.0e-16 or (1.0-r) <= 1.0e-16):
        return -1.0e32
    else:
        return mu+sigma*sqrt(2.0)*erfcinv(2.0*(1.0-r)) 

def Flat_Prior(val, minimum, maximum):
    if val<minimum or val>maximum:
        to_return = -inf
    else:
        to_return =  log(1./(maximum-minimum))
    return to_return 

def Gauss_Prior(val, xmid, xwid):
    return -0.5*((val-xmid)/xwid)**2 
  
def Prior_func(Priors, param):
    p = 0
    
    #Sum log of priors
    for i in xrange(len(Priors)):
        if Priors[i][0]=='flat':
            p += Flat_Prior(param[i], Priors[i][1], Priors[i][2])
        if Priors[i][0]=='gauss':
            p += Gauss_Prior(param[i], Priors[i][1], Priors[i][2])
            
            
    return p