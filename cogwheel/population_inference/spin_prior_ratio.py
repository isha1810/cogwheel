"""
PriorRatio classes for spin models 
"""
import numpy as np
import pandas as pd
from scipy import stats

from cogwheel.cosmology import (z_of_d_luminosity,
                                comoving_to_luminosity_diff_vt_ratio)
from cogwheel.prior import IdentityTransformMixin, Prior

from .base_prior_ratio import PriorRatio
from .jacobians import (m1_m2_to_m1s_q,
                        chieff_cumchidiff_to_s1z_s2z)
from .pdfs import (powerlaw,
                    smoothing_function,
                    smoothed_powerlaw_q,
                    truncated_gaussian)
from .pe_priors import (powerlaw_peak_primary_mass_lnp, powerlaw_primary_mass_lnp,
                        smoothed_powerlaw_peak_primary_mass_lnp,
                        powerlaw_mass_ratio_lnp, gaussian_mass_ratio_prior,
                        uniform_chieff_cartesian_spins_lnp,
                        uniform_chieff_lnp, gaussian_chieff_prior,
                        natally_spinning_prior, tidally_locked_secondary_spin_prior,
                        tidally_torqued_secondary_spin_prior,
                        lvc_lnp_cosmo)

# ----------------------------------------------------------------------
# Prior Ratio classes involving spin A2 model

class PLPA2ToLVCPriorRatio(PriorRatio):
    numerator = 'PLPA2'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'uniform_chieff_cartesian_spins_lnp',
                          'lvc_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 
                   'm_mean', 'm_std', 'beta', 'f', 'sigma_chi']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
        
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std, beta, f, sigma_chi):
        
        natal_spins_chieff = natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0)
        tidal_locked_chieff = tidally_locked_secondary_spin_prior(chieff, q, sigma_chi,
                                                 chieff_min=-1.0, chieff_max=1.0)
        spin_lnp = (np.log((1-f)*natal_spins_chieff + f*tidal_locked_chieff)
                      - uniform_chieff_lnp(max_chieff=0.998)
                      + uniform_chieff_cartesian_spins_lnp)
        
        mass_lnp = (powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                            m_min, m_max, m_mean, m_std)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_spin_lnp = mass_lnp + spin_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp


class PowerlawA2ToLVCPriorRatio(PriorRatio):
    numerator = 'PowerlawA2'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'uniform_chieff_cartesian_spins_lnp',
                          'lvc_lnp']
    hyperparams = ['alpha', 'm_min', 'm_max', 'beta', 'f', 'sigma_chi']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
        
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      alpha, m_min, m_max, beta, f, sigma_chi):
        
        natal_spins_chieff = natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0)
        tidal_locked_chieff = tidally_locked_secondary_spin_prior(chieff, q, sigma_chi,
                                                 chieff_min=-1.0, chieff_max=1.0)
        spin_lnp = (np.log((1-f)*natal_spins_chieff + f*tidal_locked_chieff)
                      - uniform_chieff_lnp(max_chieff=0.998)
                      + uniform_chieff_cartesian_spins_lnp)
        
        mass_lnp = (powerlaw_primary_mass_lnp(m1_source, alpha, m_min, m_max)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_spin_lnp = mass_lnp + spin_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp

# ----------------------------------------------------------------------
# Prior Ratio classes involving spin A4 model

class PLPA4ToLVCPriorRatio(PriorRatio):
    numerator = 'PLPA4'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'uniform_chieff_cartesian_spins_lnp',
                          'lvc_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 
                   'm_mean', 'm_std', 'beta', 'f', 'sigma_chi']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
    
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std, beta, f, sigma_chi):
        
        natal_spins_chieff = natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0)
        tidal_torqued_chieff = tidally_torqued_secondary_spin_prior(chieff, q, sigma_chi)
        
        spin_lnp = (np.log((1-f)*natal_spins_chieff + f*tidal_torqued_chieff)
                      - uniform_chieff_lnp(max_chieff=0.998)
                      + uniform_chieff_cartesian_spins_lnp)
        
        mass_lnp = (powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                            m_min, m_max, m_mean, m_std)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_spin_lnp = mass_lnp + spin_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp

class PowerlawA4ToLVCPriorRatio(PriorRatio):
    numerator = 'PowerlawA4'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'uniform_chieff_cartesian_spins_lnp', 
                          'lvc_lnp']
    hyperparams = ['alpha', 'm_min', 'm_max', 'beta', 'f', 'sigma_chi']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))

    def lnprior_ratio(self, m1_source, q, chieff,
                      z, uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      alpha, m_min, m_max, beta, f, sigma_chi):
        
        natal_spins_chieff = natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0)
        tidal_torqued_chieff = tidally_torqued_secondary_spin_prior(chieff, q, sigma_chi)
        spin_lnp = (np.log((1-f)*natal_spins_chieff + f*tidal_torqued_chieff)
                      - uniform_chieff_lnp(max_chieff=0.998)
                      + uniform_chieff_cartesian_spins_lnp)
        
        mass_lnp = (powerlaw_primary_mass_lnp(m1_source, alpha, m_min, m_max)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_spin_lnp = mass_lnp + spin_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp
    
# ----------------------------------------------------------------------
# Prior Ratio classes involving only natal spins

class PLPNatalSpinToLVCPriorRatio(PriorRatio):
    numerator = 'PLPNatalSpin'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'uniform_chieff_cartesian_spins_lnp',
                          'lvc_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 
                   'm_mean', 'm_std', 'beta', 'sigma_chi']
    
    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
    
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std, beta, sigma_chi):
        
        natal_spins_chieff_lnp = np.log(natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0))
        spin_lnp = (natal_spins_chieff_lnp 
                    - uniform_chieff_lnp(max_chieff=0.998)
                    + uniform_chieff_cartesian_spins_lnp)
        
        mass_lnp = (powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                            m_min, m_max, m_mean, m_std)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_spin_lnp = mass_lnp + spin_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp

class PowerlawNatalSpinToLVCPriorRatio(PriorRatio):
    numerator = 'PowerlawNatalSpin'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'uniform_chieff_cartesian_spins_lnp', 
                          'lvc_lnp']
    hyperparams = ['alpha', 'm_min', 'm_max', 'beta', 'sigma_chi']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
    
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      alpha, m_min, m_max, beta, sigma_chi):
        
        natal_spins_chieff_lnp = np.log(natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0))
        spin_lnp = (natal_spins_chieff_lnp
                    - uniform_chieff_lnp(max_chieff=0.998)
                    + uniform_chieff_cartesian_spins_lnp)
        
        mass_lnp = (powerlaw_primary_mass_lnp(m1_source, alpha, m_min, m_max)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_spin_lnp = mass_lnp + spin_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp    
# ----------------------------------------------------------------------
# Prior Ratio classes involving uniform+delta function spin model

class DeltaUniformQChieffToLVCPriorRatio(PriorRatio):
    numerator = 'DeltaUniformQChieff'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'truncnorm_chieff_zero',
                          'uniform_chieff_cartesian_spins_lnp',
                          'lvc_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 'm_mean', 'm_std', 
                   'beta', 'f']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        samples['truncnorm_chieff_zero'] = gaussian_chieff_prior(samples['chieff'],
                                                               0.0, 0.05)
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
        
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, truncnorm_chieff_zero, 
                      uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std,
                      beta, f):
        max_spin=0.998
        q_min = m_min/m1_source
        truncnorm_q_one = gaussian_mass_ratio_lnp(q, 1.0, 0.05, q_min.values, 1.0)
        uniform_chieff_prior = 1/(2*max_spin)
        q_min = m_min/m1_source
        spin_lnp = (np.log(f*truncnorm_chieff_zero*truncnorm_q_one + 
                           (1-f)*powerlaw(q.values, beta, q_min.values, 1.0)*uniform_chieff)
                    - uniform_chieff_lnp(max_chieff=0.998)
                    + uniform_chieff_cartesian_spins_lnp)
                           
        mass_lnp = powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                        m_min, m_max, m_mean, m_std)
        
        mass_spin_lnp = mass_lnp+spin_lnp
        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)

        return pop_lnp-lvc_lnp

class SmoothedDeltaUniformQChieffToLVCPriorRatio(PriorRatio):
    numerator = 'SmoothedDeltaUniformQChieff'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'truncnorm_chieff_zero',
                          'uniform_chieff_cartesian_spins_lnp',
                          'lvc_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 'm_mean', 'm_std', 
                   'beta', 'delta_m', 'f']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        samples['truncnorm_chieff_zero'] = gaussian_chieff_prior(samples['chieff'],
                                                               0.0, 0.05)
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
        
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, truncnorm_chieff_zero, 
                      uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std,
                      beta, delta_m, f):
        max_spin=0.998
        q_min = m_min/m1_source
        truncnorm_q_one = gaussian_mass_ratio_lnp(q, 1.0, 0.05, q_min.values, 1.0)
        uniform_chieff = 1/(2*max_spin)
        spin_lnp = (np.log(f*truncnorm_chieff_zero*truncnorm_q_one + 
                        (1-f)*smoothed_powerlaw_q(q, 
                            m1_source, beta, m_min, m_max, delta_m)*uniform_chieff)
                    - uniform_chieff_lnp(max_chieff=0.998)
                    + uniform_chieff_cartesian_spins_lnp)
                           
        mass_lnp = smoothed_powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                        m_min, m_max, m_mean, m_std, delta_m)
        
        mass_spin_lnp = mass_lnp+spin_lnp
        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)

        return pop_lnp-lvc_lnp

# ----------------------------------------------------------------------
# Prior Ratio class for PLP+Gaussian Chieff model

class PLPGaussianChieffToLVCPriorRatio(PriorRatio):
    numerator = 'PLPGaussianChieff'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'uniform_chieff_cartesian_spins_lnp',
                          'lvc_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 
                   'm_mean', 'm_std', 'beta', 'mu_chi', 'sigma_chi']
    
    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
    
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std, beta, mu_chi, sigma_chi):
        
        gaussian_chieff_lnp = np.log(gaussian_chieff_prior(chieff,
                                                    mu_chi, sigma_chi))
        spin_lnp = (gaussian_chieff_lnp 
                    - uniform_chieff_lnp(max_chieff=0.998)
                    + uniform_chieff_cartesian_spins_lnp)
        
        mass_lnp = (powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                            m_min, m_max, m_mean, m_std)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_spin_lnp = mass_lnp + spin_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp

# ----------------------------------------------------------------------
# Prior Ratio class for components of PLP+A2 model

class NatallySpinningComponentToLVCPrior(PriorRatio):
    numerator = 'NatallySpinningComponent'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'uniform_chieff_cartesian_spins_lnp',
                          'lvc_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 
                   'm_mean', 'm_std', 'beta', 'f', 'sigma_chi']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
        
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std, beta, f, sigma_chi):
        
        natal_spins_chieff = natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0)
        spin_lnp = (np.log((1-f)*natal_spins_chieff)
                      - uniform_chieff_lnp(max_chieff=0.998)
                      + uniform_chieff_cartesian_spins_lnp)
        
        mass_lnp = (powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                            m_min, m_max, m_mean, m_std)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_spin_lnp = mass_lnp + spin_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp

class TidallyLockedComponentToLVCPrior(PriorRatio):
    numerator = 'TidallyLockedComponent'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'uniform_chieff_cartesian_spins_lnp',
                          'lvc_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 
                   'm_mean', 'm_std', 'beta', 'f', 'sigma_chi']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
        
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std, beta, f, sigma_chi):
        
        tidal_locked_chieff = tidally_locked_secondary_spin_prior(chieff, q, sigma_chi,
                                                 chieff_min=-1.0, chieff_max=1.0)
        spin_lnp = (np.log(f*tidal_locked_chieff)
                      - uniform_chieff_lnp(max_chieff=0.998)
                      + uniform_chieff_cartesian_spins_lnp)
        
        mass_lnp = (powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                            m_min, m_max, m_mean, m_std)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_spin_lnp = mass_lnp + spin_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp

class TidallyTorquedComponentToLVCPriorRatio(PriorRatio):
    numerator = 'TidallyTorquedComponent'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'uniform_chieff_cartesian_spins_lnp',
                          'lvc_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 
                   'm_mean', 'm_std', 'beta', 'f', 'sigma_chi']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['uniform_chieff_cartesian_spins_lnp'] = \
                uniform_chieff_cartesian_spins_lnp(
                    samples['s1x'], samples['s1y'], samples['s1z'], 
                    samples['s2x'], samples['s2y'], samples['s2z'],
                    samples['q'])
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
    
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, uniform_chieff_cartesian_spins_lnp, lvc_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std, beta, f, sigma_chi):
        
        tidal_torqued_chieff = tidally_torqued_secondary_spin_prior(chieff, q, sigma_chi)
        
        spin_lnp = (np.log(f*tidal_torqued_chieff)
                      - uniform_chieff_lnp(max_chieff=0.998)
                      + uniform_chieff_cartesian_spins_lnp)
        
        mass_lnp = (powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                            m_min, m_max, m_mean, m_std)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_spin_lnp = mass_lnp + spin_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp

# ----------------------------------------------------------------------
# Prior Ratio class for components of PLP+natal spins model

# class PLPNatalSpinCartesianCoordsToLVCPriorRatio(PriorRatio):
#     numerator = 'PLPNatalSpinCartesianCoords'
#     denominator = 'LVCPrior'
#     params = ['m1_source','q', 's1z', 's2z']
#     base_quantities = ['d_luminosity']
#     derived_quantities = ['z', 'lvc_lnp']
#     hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 
#                    'm_mean', 'm_std', 'beta', 'sigma_chi']
    
#     def compute_auxiliary_quantities(self, samples):
#         samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
#         max_spin=0.998
#         samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
#                                           samples['s2x'], samples['s2y'], samples['s2z'])
#                               + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
    
#     def lnprior_ratio(self, m1_source, q, chieff,
#                       z, lvc_lnp,
#                       lambda_peak, alpha, m_min, m_max, m_mean, m_std, beta, sigma_chi):

#         natal_spin_s1z_lnp = np.log(truncated_gaussian(s1z,-1.0, 1.0, 0.0, sigma_chi))
#         natal_spin_s2z_lnp = np.log(truncated_gaussian(s2z,-1.0, 1.0, 0.0, sigma_chi))
#         inplane_spins_s1 = -np.log(np.pi(1-s1z**2))
#         inplane_spins_s2 = -np.log(np.pi(1-s2z**2))
#         spin_lnp = (natal_spin_s1z_lnp + inplane_spins_s1 +
#                            natal_spin_s2z_lnp + inplane_spins_s2)
        
#         mass_lnp = (powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
#                             m_min, m_max, m_mean, m_std)
#                          + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
#         mass_spin_lnp = mass_lnp + spin_lnp

#         time_dilation = -np.log(1+z)
#         pop_lnp = (mass_spin_lnp
#                    + time_dilation)
#         return pop_lnp-lvc_lnp