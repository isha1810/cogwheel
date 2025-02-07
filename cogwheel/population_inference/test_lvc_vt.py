import numpy as np
import pandas as pd
from scipy import stats

from cogwheel.cosmology import (z_of_d_luminosity,
                                comoving_to_luminosity_diff_vt_ratio)

from .base_prior_ratio import PriorRatio
from .pdfs import powerlaw

class TestPriorToLVCPriorRatio(PriorRatio):
    numerator = 'TestPrior'
    denominator = 'LVCPrior'
    params = ['m1_source','m2_source']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z']
    hyperparams = ['m1_mean', 'm2_mean', 'sig_lognorm']

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity']) 

    def lnprior_ratio(self, m1_source, m2_source, z,
                      m1_mean, m2_mean, sig_lognorm):
        
        logc = -stats.lognorm.logcdf(m1_source, sig_lognorm, scale=m2_mean) 
        m2_source_lnp = np.where(m2_source<m1_source, stats.lognorm.logpdf(m2_source, sig_lognorm, scale=m2_mean) + logc, np.NINF)
        m1_source_lnp = stats.lognorm.logpdf(m1_source, sig_lognorm, scale=m1_mean)
        test_mass_lnp = m2_source_lnp + m1_source_lnp
        
        time_dilation = - np.log(1+z)
        test_lnp = test_mass_lnp + time_dilation

        lvc_mass_jacobian = 2*np.log(1+z)
        lvc_mass_lnp = lvc_mass_jacobian
        lvc_lnp = lvc_mass_lnp
        
        return test_lnp - lvc_lnp

def uniform_chieff_s1_s2(q, chieff, s1z, s2z,
                          chieff_max, chieff_min):
    '''
    uniform in chieff and chidiff in s1z, s2z coordinates
    '''
    smax=0.998
    
    # P_chieff = 1/(chieff_max-chieff_min)
    prob_s1x_s1y = 1/(smax**2-s1z**2)
    prob_s2x_s2y = 1/(smax**2-s2z**2)

    delta = (1-q)/(1+q)
    log_Ps1s2 = np.log(prob_s1x_s1y) + np.log(prob_s2x_s2y)
    mask = abs(chieff)>smax*delta
    log_Ps1s2[mask] += (np.log(1-delta) - np.log(1 - abs(chieff)/smax))[mask]
    
    log_ps1_ps2 = np.where((chieff>chieff_min) & (chieff<chieff_max), log_Ps1s2, np.NINF)
   
    return log_ps1_ps2

def logprob_spin_cartesian_coord_ajit(q, chieff, s1z, s2z,
                          chieff_max, chieff_min):
    
    smax = 0.998
    
    prob_s1x_s1y = 1/(smax**2-s1z**2)/np.pi
    prob_s2x_s2y = 1/(smax**2-s2z**2)/np.pi
    
    P_s1s2 = prob_s1x_s1y*prob_s2x_s2y
    mask = abs(chieff)>smax*(1-q)/(1+q)
    P_s1s2[mask] *= (2*q/(1+q)/(1-abs(chieff)/smax))[mask]
    
    log_ps1_ps2 = np.where((chieff>chieff_min) & (chieff<chieff_max), np.log(P_s1s2), np.NINF)
    
    return log_ps1_ps2

class QChieffLogNormToLVCPriorRatio(PriorRatio):
    numerator = 'QChieffLogNorm'
    denominator = 'LVCPrior'
    params = ['m1_source', 'q','chieff', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z']
    hyperparams = ['q_min', 'q_max', 'chieff_min', 'chieff_max', 'm1min', 'm1max']

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity']) 
        # if 'lvc_lnp' not in samples.keys():
        #     s1x, s1y, s1z = samples['s1x'], samples['s1y'], samples['s1z']
        #     s2x, s2y, s2z = samples['s2x'], samples['s2y'], samples['s2z']
        #     q=samples['q']
        #     z=samples['z']
        #     m1_source=samples['m1_source']
        #     max_spin=0.998
        #     # lvc_spin_jacobian = np.log(2*max_spin) + np.log(1+q) - np.log(q)
        #     lvc_spin_lnp = (- np.log(4*np.pi*(s1x**2 + s1y**2 + s1z**2)*max_spin)
        #                       - np.log(4*np.pi*(s2x**2 + s2y**2 + s2z**2)*max_spin)
        #                       # + lvc_spin_jacobian
        #                    )
        #     lvc_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        #     lvc_mass_lnp = lvc_mass_jacobian
        #     samples['lvc_lnp'] = lvc_spin_lnp + lvc_mass_lnp
            
        if 'pop_inplane_spin_lnp' not in samples.keys():
            s1x, s1y, s1z = samples['s1x'], samples['s1y'], samples['s1z']
            s2x, s2y, s2z = samples['s2x'], samples['s2y'], samples['s2z']
            q=samples['q']
            z=samples['z']
            m1_source=samples['m1_source']
            max_inplane_spin1 = np.sqrt(0.998**2 - s1z**2)
            max_inplane_spin2 = np.sqrt(0.998**2 - s2z**2)
            inplane_spin1 = np.sqrt(s1x**2 + s1y**2)
            inplane_spin2 = np.sqrt(s2x**2 + s2y**2)
            samples['pop_inplane_spin_lnp'] = (-np.log(2*np.pi*inplane_spin1*max_inplane_spin1)
                                   -np.log(2*np.pi*inplane_spin2*max_inplane_spin2))
            
    def lnprior_ratio(self, m1_source, q, chieff, s1x, s1y, s1z, s2x, s2y, s2z, z,
                      q_min, q_max, chieff_min, chieff_max, m1min, m1max):

        alpha1=-2.35
        mmin=2.0
        mmax=100.0
        m1_source_lnp = np.log(powerlaw(m1_source, alpha1, m1min, m1max))
        min_allowed_q = mmin/m1_source
        q_min = np.where(q_min<min_allowed_q, min_allowed_q, q_min)
 
        q_lnp = np.where(np.logical_and(q>q_min, q<q_max), -np.log(q_max-q_min), np.NINF)
        mass_lnp = m1_source_lnp + q_lnp
        
        # chieff_ln_norm = - np.log(chieff_max-chieff_min)
        # spin_lnp = uniform_chieff_s1_s2(q, chieff, s1z, s2z, chieff_max, chieff_min)
        max_spin=0.998

        spin_lnp = logprob_spin_cartesian_coord_ajit(q, chieff, s1z, s2z, chieff_max, chieff_min)
        
        time_dilation = - np.log(1+z)
        pop_lnp = mass_lnp + spin_lnp + time_dilation

        lvc_spin_lnp = (- np.log(4*np.pi*(s1x**2 + s1y**2 + s1z**2)*max_spin)
                          - np.log(4*np.pi*(s2x**2 + s2y**2 + s2z**2)*max_spin))
        lvc_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        lvc_mass_lnp = lvc_mass_jacobian
        lvc_lnp2 = lvc_spin_lnp + lvc_mass_lnp
        
        return pop_lnp - lvc_lnp2
        