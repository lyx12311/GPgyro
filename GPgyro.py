import jax
import numpy as np
import jax.numpy as jnp
from tinygp import kernels
import pandas as pd

import jaxopt
from tinygp import GaussianProcess, kernels, transforms
from functools import partial
import arviz as az
import corner

path_to_files = './'

def mean_function_both(params, X):
    # Prot broken low
    teffnorm = (7000.-X[0])/(7000.-params["teff_cut"])
    prot = jnp.power(10.,X[1])
    # Prot broken low
    stepfunc_low_prot = 1.0 / (1.0 + jnp.exp(-(jnp.log10(params['prot_cut']) - X[1]) / abs(params['w_prot'])))
    stepfunc_high_prot = 1.0 / (1.0 + jnp.exp(-(-jnp.log10(params['prot_cut']) + X[1]) / abs(params['w_prot'])))
    
    mod_high_prot = jnp.power(prot,params["b"])
    mod_high_prot = mod_high_prot*stepfunc_high_prot
    
    mod_low_prot = jnp.power(prot,params["b2"])*jnp.power(params["prot_cut"],params["b"]-params["b2"])
    mod_low_prot = mod_low_prot*stepfunc_low_prot
    
    prot_func = mod_high_prot+mod_low_prot
    
    # teff broken low
    stepfunc_high_teff = 1.0 / (1.0 + jnp.exp(-(1. - teffnorm) / abs(params['w_teff'])))
    stepfunc_low_teff = 1.0 / (1.0 + jnp.exp(-(-1. + teffnorm) / abs(params['w_teff'])))
    
    
    mod_high_teff = jnp.power(teffnorm-params["c"],params["d"])
    mod_high_teff = mod_high_teff*stepfunc_high_teff
    
    mod_low_teff = jnp.power(teffnorm-params["c"],params["d2"])*jnp.power(1.-params["c"],params["d"]-params["d2"])
    mod_low_teff = mod_low_teff*stepfunc_low_teff
    teff_func = mod_high_teff+mod_low_teff
    
    return params["a"]*prot_func*teff_func


def build_gp(params, X, yerr):
    kernel = jnp.exp(params["log_amp"]) * transforms.Linear(
        jnp.array([jnp.exp(-params["log_scale1"]), jnp.exp(-params["log_scale2"])]),
        kernels.ExpSquared()
    )

    return GaussianProcess(
        kernel, X, diag=yerr**2,mean=partial(mean_function_both, params)
    )



def mean_function_both_FC(params, X):
    # Prot broken low
    teffnorm = (3500.-X[0])/500.
    prot = jnp.power(10.,X[1])
    # Prot broken low
    stepfunc_low_prot = 1.0 / (1.0 + jnp.exp(-(jnp.log10(params['prot_cut']) - X[1]) / abs(params['w_prot'])))
    stepfunc_high_prot = 1.0 / (1.0 + jnp.exp(-(-jnp.log10(params['prot_cut']) + X[1]) / abs(params['w_prot'])))
    
    mod_high_prot = jnp.power(prot,params["b"])
    mod_high_prot = mod_high_prot*stepfunc_high_prot
    
    mod_low_prot = jnp.power(prot,params["b2"])*jnp.power(params["prot_cut"],params["b"]-params["b2"])
    mod_low_prot = mod_low_prot*stepfunc_low_prot
    
    prot_func = mod_high_prot+mod_low_prot
    
    # teff broken low
    mod_high_teff = jnp.power(teffnorm-params["c"],params["d"])
    teff_func = mod_high_teff
    
    return params["a"]*prot_func*teff_func


def build_gp_FC(params, X, yerr):
    kernel = jnp.exp(params["log_amp"]) * transforms.Linear(
        jnp.array([jnp.exp(-params["log_scale1"]), jnp.exp(-params["log_scale2"])]),
        kernels.ExpSquared()
    )

    return GaussianProcess(
        kernel, X, diag=yerr**2,mean=partial(mean_function_both_FC, params)
    )

allkeys = ['a', 'b', 'b2', 'c', 'd', 'd2', 'log_amp', 'log_scale1',
           'log_scale2', 'prot_cut', 'teff_cut', 'w_prot', 'w_teff']
sample_PC = np.load(path_to_files+'sample_PC.npy')

allkeys_fc = ['a', 'b', 'b2', 'c', 'd', 'log_amp', 'log_scale1',
           'log_scale2', 'prot_cut', 'w_prot']
sample_FC = np.load(path_to_files+'sample_FC.npy')

sample_pc_fig = np.zeros(np.shape(sample_FC))
sample_pc_fig[0:5,:] = sample_PC[0:5,:]
sample_pc_fig[5:9,:] = sample_PC[6:10,:]
sample_pc_fig[9,:] = sample_PC[11,:]


m = ((sample_FC[6,:]>5)&(sample_FC[6,:]<10))
sample_FC = sample_FC[:,m]
sample_PC = pd.DataFrame(sample_PC.T, columns=allkeys)
sample_PC = sample_PC.sample(n=100)

def GP_gyro_PC(X, sample, filepath=path_to_files):
    X_t = np.load(filepath+'X.npy')
    y_t = np.load(filepath+'yerr.npy')
    y = np.load(filepath+'y.npy')
    allkeys = ['a', 'b', 'b2', 'c', 'd', 'd2', 'log_amp', 'log_scale1',
           'log_scale2', 'prot_cut', 'teff_cut', 'w_prot', 'w_teff']
    outputage = np.zeros((len(sample),len(X)))
    for i in trange(len(sample)):
        val = sample.iloc[i][allkeys]
        uncorr_gp = build_gp(val, X_t, y_t)
        outputage[i,:] = uncorr_gp.condition(y, X).gp.loc.reshape(len(X))
    #print(outputage)
    ages = np.zeros(len(X))
    ages_p = np.zeros(len(X))
    ages_m = np.zeros(len(X))
    for i in range(len(X)):
        mcmc = np.percentile((outputage)[:, i][(outputage)[:, i]==(outputage)[:, i]], [16, 50, 84])
        q = np.diff(mcmc)
        ages[i] = mcmc[1]
        ages_m[i] = -q[0]
        ages_p[i] = q[1]
    return ages, ages_m, ages_p


sample_FC = pd.DataFrame(sample_FC.T, columns=allkeys_fc)
sample_FC = sample_FC.sample(n=100)

def GP_gyro_FC(X, sample):
    X_t = np.load(filepath+'X_fc.npy')
    y_t = np.load(filepath+'yerr_fc.npy')
    y = np.load(filepath+'y_fc.npy')
    allkeys = ['a', 'b', 'b2', 'c', 'd', 'log_amp', 'log_scale1',
           'log_scale2', 'prot_cut', 'w_prot']
    outputage = np.zeros((len(sample),len(X)))
    for i in trange(len(sample)):
        val = sample.iloc[i][allkeys]
        uncorr_gp = build_gp_FC(val, X_t, y_t)
        outputage[i,:] = uncorr_gp.condition(y, X).gp.loc.reshape(len(X))
        #print(outputage[i,:])
    ages = np.zeros(len(X))
    ages_p = np.zeros(len(X))
    ages_m = np.zeros(len(X))
    for i in range(len(X)):
        try:
            mcmc = np.percentile((outputage)[:, i][(outputage)[:, i]==(outputage)[:, i]], [16, 50, 84])
            q = np.diff(mcmc)
            ages[i] = mcmc[1]
            ages_m[i] = -q[0]
            ages_p[i] = q[1]
        except:
            ages[i], ages_m[i], ages_p[i] = np.nan, np.nan, np.nan
    return ages, ages_m, ages_p


def GP_gyro(X, MG):
    jaogap = fitpoints([3560/2+3526.5/2, 3427.36/2+3395.1/2], [10.09, 10.24])
    m_pc = (MG<jaogap(X[:,0]))
    ages_all = np.zeros(len(X))
    ages_p_all = np.zeros(len(X))
    ages_m_all = np.zeros(len(X))
    
    ages_all[m_pc], ages_p_all[m_pc], ages_m_all[m_pc] = GP_gyro_PC(X[m_pc,:], sample_PC)
    ages_all[~m_pc], ages_p_all[~m_pc], ages_m_all[~m_pc] = GP_gyro_FC(X[~m_pc,:], sample_FC)
    return ages_all, ages_m_all, ages_p_all
