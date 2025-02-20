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
                    truncated_gaussian)
from .pe_priors import (powerlaw_peak_primary_mass_lnp, powerlaw_primary_mass_lnp,
                        powerlaw_mass_ratio_lnp,
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
    derived_quantities = ['z', 'pop_inplane_spin_lnp',
                          'pop_spin_jacobian_lnp', 'lvc_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 
                   'm_mean', 'm_std', 'beta', 'f', 'sigma_chi']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['pop_inplane_spin_lnp'] = (-np.log(np.pi*(max_spin**2-samples['s1z']**2))
                                           -np.log(np.pi*(max_spin**2-samples['s2z']**2)))
        samples['pop_spin_jacobian_lnp'] = chieff_cumchidiff_to_s1z_s2z(
                                            samples['chieff'], samples['q']
                                            , max_spin=0.998)
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
        
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, pop_inplane_spin_lnp, pop_spin_jacobian_lnp, lvc_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std, beta, f, sigma_chi):
        
        natal_spins_chieff = natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0)
        tidal_locked_chieff = tidally_locked_secondary_spin_prior(chieff, q, sigma_chi,
                                                 chieff_min=-1.0, chieff_max=1.0)
        chieff_lnp = (np.log((1-f)*natal_spins_chieff + f*tidal_locked_chieff)
                      + pop_spin_jacobian_lnp)
        
        mass_lnp = (powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                            m_min, m_max, m_mean, m_std)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_q_chieff_lnp = mass_lnp + chieff_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_q_chieff_lnp + pop_inplane_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp


class PowerlawA2ToLVCPriorRatio(PriorRatio):
    numerator = 'PowerlawA2'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'pop_inplane_spin_lnp',
                          'pop_spin_jacobian_lnp', 'lvc_lnp']
    hyperparams = ['alpha', 'm_min', 'm_max', 'beta', 'f', 'sigma_chi']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['pop_inplane_spin_lnp'] = (-np.log(np.pi*(max_spin**2-samples['s1z']**2))
                                           -np.log(np.pi*(max_spin**2-samples['s2z']**2)))
        samples['pop_spin_jacobian_lnp'] = chieff_cumchidiff_to_s1z_s2z(
                                            samples['chieff'], samples['q']
                                            , max_spin=0.998)
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
        
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, pop_inplane_spin_lnp, pop_spin_jacobian_lnp, lvc_lnp,
                      alpha, m_min, m_max, beta, f, sigma_chi):
        
        natal_spins_chieff = natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0)
        tidal_locked_chieff = tidally_locked_secondary_spin_prior(chieff, q, sigma_chi,
                                                 chieff_min=-1.0, chieff_max=1.0)
        chieff_lnp = (np.log((1-f)*natal_spins_chieff + f*tidal_locked_chieff)
                      + pop_spin_jacobian_lnp)
        
        mass_lnp = (powerlaw_primary_mass_lnp(m1_source, alpha, m_min, m_max)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_q_chieff_lnp = mass_lnp + chieff_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_q_chieff_lnp + pop_inplane_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp

# ----------------------------------------------------------------------
# Prior Ratio classes involving spin A4 model

class PLPA4ToLVCPriorRatio(PriorRatio):
    numerator = 'PLPA4'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'pop_inplane_spin_lnp',
                          'pop_spin_jacobian_lnp', 'lvc_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 
                   'm_mean', 'm_std', 'beta', 'f', 'sigma_chi']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['pop_inplane_spin_lnp'] = (-np.log(np.pi*(max_spin**2-samples['s1z']**2))
                                           -np.log(np.pi*(max_spin**2-samples['s2z']**2)))
        samples['pop_spin_jacobian_lnp'] = chieff_cumchidiff_to_s1z_s2z(
                                            samples['chieff'], samples['q']
                                            , max_spin=0.998)
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
    
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, pop_inplane_spin_lnp, pop_spin_jacobian_lnp, lvc_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std, beta, f, sigma_chi):
        
        natal_spins_chieff = natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0)
        tidal_torqued_chieff = tidally_torqued_secondary_spin_prior(chieff, q, sigma_chi)
        chieff_lnp = (np.log((1-f)*natal_spins_chieff + f*tidal_torqued_chieff)
                      + pop_spin_jacobian_lnp)
        
        mass_lnp = (powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                            m_min, m_max, m_mean, m_std)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_q_chieff_lnp = mass_lnp + chieff_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_q_chieff_lnp + pop_inplane_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp

class PowerlawA4ToLVCPriorRatio(PriorRatio):
    numerator = 'PowerlawA4'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'pop_inplane_spin_lnp',
                          'pop_spin_jacobian_lnp', 'lvc_lnp']
    hyperparams = ['alpha', 'm_min', 'm_max', 'beta', 'f', 'sigma_chi']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['pop_inplane_spin_lnp'] = (-np.log(np.pi*(max_spin**2-samples['s1z']**2))
                                           -np.log(np.pi*(max_spin**2-samples['s2z']**2)))
        samples['pop_spin_jacobian_lnp'] = chieff_cumchidiff_to_s1z_s2z(
                                            samples['chieff'], samples['q']
                                            , max_spin=0.998)
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))

    def lnprior_ratio(self, m1_source, q, chieff,
                      z, pop_inplane_spin_lnp, pop_spin_jacobian_lnp, lvc_lnp,
                      alpha, m_min, m_max, beta, f, sigma_chi):
        
        natal_spins_chieff = natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0)
        tidal_torqued_chieff = tidally_torqued_secondary_spin_prior(chieff, q, sigma_chi)
        chieff_lnp = (np.log((1-f)*natal_spins_chieff + f*tidal_torqued_chieff)
                      + pop_spin_jacobian_lnp)
        
        mass_lnp = (powerlaw_primary_mass_lnp(m1_source, alpha, m_min, m_max)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_q_chieff_lnp = mass_lnp + chieff_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_q_chieff_lnp + pop_inplane_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp
    
# ----------------------------------------------------------------------
# Prior Ratio classes involving only natal spins

class PLPNatalSpinToLVCPriorRatio(PriorRatio):
    numerator = 'PLPNatalSpin'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'pop_inplane_spin_lnp',
                          'pop_spin_jacobian_lnp', 'lvc_lnp']
    hyperparams = ['lambda_peak', 'alpha', 'm_min', 'm_max', 
                   'm_mean', 'm_std', 'beta', 'sigma_chi']
    
    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['pop_inplane_spin_lnp'] = (-np.log(np.pi*(max_spin**2-samples['s1z']**2))
                                           -np.log(np.pi*(max_spin**2-samples['s2z']**2)))
        samples['pop_spin_jacobian_lnp'] = chieff_cumchidiff_to_s1z_s2z(
                                            samples['chieff'], samples['q']
                                            , max_spin=0.998)
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
    
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, pop_inplane_spin_lnp, pop_spin_jacobian_lnp, lvc_lnp,
                      lambda_peak, alpha, m_min, m_max, m_mean, m_std, beta, sigma_chi):
        
        natal_spins_chieff_lnp = np.log(natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0))
        chieff_lnp = (natal_spins_chieff_lnp + pop_spin_jacobian_lnp)
        
        mass_lnp = (powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                            m_min, m_max, m_mean, m_std)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_q_chieff_lnp = mass_lnp + chieff_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_q_chieff_lnp + pop_inplane_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp

class PowerlawNatalSpinToLVCPriorRatio(PriorRatio):
    numerator = 'PowerlawNatalSpin'
    denominator = 'LVCPrior'
    params = ['m1_source','q', 'chieff']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'pop_inplane_spin_lnp',
                          'pop_spin_jacobian_lnp', 'lvc_lnp']
    hyperparams = ['alpha', 'm_min', 'm_max', 'beta', 'sigma_chi']

    def compute_auxiliary_quantities(self, samples):
        samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        max_spin=0.998
        samples['pop_inplane_spin_lnp'] = (-np.log(np.pi*(max_spin**2-samples['s1z']**2))
                                           -np.log(np.pi*(max_spin**2-samples['s2z']**2)))
        samples['pop_spin_jacobian_lnp'] = chieff_cumchidiff_to_s1z_s2z(
                                            samples['chieff'], samples['q']
                                            , max_spin=0.998)
        samples['lvc_lnp'] = (lvc_lnp_cosmo(samples['s1x'], samples['s1y'], samples['s1z'],
                                          samples['s2x'], samples['s2y'], samples['s2z'])
                              + m1_m2_to_m1s_q(samples['m1_source'], samples['z']))
    
    def lnprior_ratio(self, m1_source, q, chieff,
                      z, pop_inplane_spin_lnp, pop_spin_jacobian_lnp, lvc_lnp,
                      alpha, m_min, m_max, beta, sigma_chi):
        
        natal_spins_chieff_lnp = np.log(natally_spinning_prior(chieff, q, sigma_chi,
                                               chieff_min=-1.0, chieff_max=1.0))
        chieff_lnp = (natal_spins_chieff_lnp + pop_spin_jacobian_lnp)
        
        mass_lnp = (powerlaw_primary_mass_lnp(m1_source, alpha, m_min, m_max)
                         + powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min))
        mass_q_chieff_lnp = mass_lnp + chieff_lnp

        time_dilation = -np.log(1+z)
        pop_lnp = (mass_q_chieff_lnp + pop_inplane_spin_lnp
                   + time_dilation)
        return pop_lnp-lvc_lnp    

# ----------------------------------------------------------------------
# Prior Ratio classes involving uniform+delta function spin model

# class DeltaUniformQChieffToLVCPriorRatio(PriorRatio):
#     numerator = ''
#     denominator = 'LVCPrior'
#     params = ['m1_source','q', ]
#     base_quantities = ['d_luminosity']
#     derived_quantities = ['z', 'comoving_to_luminosity']
#     hyperparams = ['lambda_peak', 'alpha', 'm_max', 'm_mean', 'm_std', 'beta_q', 'm_min', 'delta_m']

