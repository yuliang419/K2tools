import emcee
from numpy import *
import matplotlib.pyplot as plt 
import model_transits
from mcmc_priors import Prior_func, Flat_Prior_dis, Gauss_Prior_dis
import corner

def logp(x, priors_info):
	return Prior_func(priors_info, x)

def transit_likelihoodfunc(params,period,times,data,sigma,priors_info):
	# PARAMS:
	# params[0, 1, 2, 3, 5] = ['T0', 'b', 'R/a', 'Rp/Rs', 'Fstar']  , for i = 0 ... len(a)-2 ---- the last ones are the limb darkening, so len(params) = 1 + len(data)
	# params[len(params)-1] = ['q1','q2']

	u1_param = params[-2]
	u2_param = params[-1]

	# check physical boundaries for LD parameters and return -inf if they exceed those boundaries
	if ( (u1_param + u2_param) > 1.) or ( u1_param < 0. ) or ( u2_param < 0. ):
		print 'unphysical limb darkening coeffs', u1_param, u2_param
		return -inf

	if (params[3]<0):
		print 'Rp/Rs<0'
		return -inf

	if (params[2]<0):
		print 'Rs/a<0'
		return -inf

	params[1] = abs(params[1])

	# calculate the likelihoods for all different transit models and sum them together
	log_likelihood = []
	prior = []

	prior.append( logp(array([params[0],params[1],params[2],params[3],params[4],u1_param,u2_param]),priors_info) ) # the prior has to be evaluated for q1 and q2, NOT u1 and u2
		
	if not isfinite(prior[0]):
	  	return -inf

	new_params = array([params[0],params[1],params[2],params[3],params[4],u1_param,u2_param])
	transit_model = model_transits.modeltransit(new_params,model_transits.occultquad,period,times)
	log_likelihood.append( -sum( ((array(transit_model)-array(data))**2. / ((2.**0.5)*array(sigma)**2) ) + (log(array(sigma)) * (2.*pi)**0.5  ) ) )

	#print 'prior / likelihood'
	#print prior
	#print log_likelihood
	return sum(array(prior)) + sum(array(log_likelihood))

def run_mcmc(epic,transit_params,period,depth,times,data,sigma,nwalkers=100,nthread=1,burnintime=300,iterations=500,saveacc=500,saveval=500,thin=100):
	# Parameters are [T0,b,R_over_a,Rp_over_Rstar,flux_star,gamma1,gamma2]
	T0_prior = ['gauss', 0, 0.05] # should be around 0.
	b_prior = ['flat', -1, 1]
	R_over_a_prior = ['gauss', 0.5, 0.1]
	Rp_over_Rstar_prior = ['flat',1e-4, 0.2]
	flux_star_prior = ['gauss', 1, 0.0002] # should be normalised to one #FIXME
	gamma1_prior = ['flat',0.01,0.7]
	gamma2_prior = ['flat',0.01,0.7]

	priors_info = [T0_prior, b_prior, R_over_a_prior, Rp_over_Rstar_prior, flux_star_prior, gamma1_prior, gamma2_prior] 
	ndim = 7

	# Initialize sampler:
  	sampler = emcee.EnsembleSampler(nwalkers, ndim, transit_likelihoodfunc, args=[period,times,data,sigma,priors_info], threads=nthread)
  	starting_pos = [transit_params + 1e-6*random.randn(ndim) for i in range(nwalkers)] # all start out in more or less the location of our best guess.
  	pos,prob,state = sampler.run_mcmc(starting_pos,1000)

	samples = sampler.chain[:, burnintime:, :].reshape((-1, ndim))
	fig = corner.corner(samples,labels=['t0','b','R/a','Rp/Rs','F_norm','gamma1','gamma2'])
	fig.savefig('outputs/'+epic+'_triangle.png')

	parnames = ['t0','b','R/a','Rp/Rs','F','gamma1','gamma2']
	for i in range(0,7):
		print parnames[i], percentile(array(samples[:,i]),[(100-68.3)/2,50,50+68.3/2])

	params = median(samples,axis=0)
	model = model_transits.modeltransit(params,model_transits.occultquad,period,times)
	plt.close('all')
	fig = plt.figure(figsize=(10,5))
	plt.plot(times*24,data,lw=0,marker='.')
	plt.plot(times*24,model,color='r')
	plt.xlabel('Time from midtransit (hours)')
	plt.ylabel('Relative flux')
	plt.savefig('outputs/'+epic+'final.pdf',dpi=150)

	return params 

