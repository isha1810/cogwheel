import numpy as np
import pandas as pd
from scipy import stats

from cogwheel.cosmology import (z_of_d_luminosity,
                                comoving_to_luminosity_diff_vt_ratio)
from cogwheel.prior import IdentityTransformMixin, Prior

from .base_prior_ratio import PriorRatio
from .pdfs import (powerlaw,
                    smoothing_function,
                    truncated_gaussian)

# class TruncatedMassModelToVolumetricPrior(PriorRatio):
#     numerator = 'TruncatedMassModel'
#     denominator = 'VolumetricPrior'
#     params = ['m1_source','q']
#     base_quantities = ['d_luminosity']
#     derived_quantities = ['z', 'comoving_to_luminosity']
#     hyperparams = ['alpha', 'm_min', 'm_max', 'beta_q']

#     def compute_auxiliary_quantities(self, d_luminosity):
#         aux_quantities_dataframe = pd.DataFrame()
#         aux_quantities_dataframe['z'] = z_of_d_luminosity(d_luminosity)
#         aux_quantities_dataframe['comoving_to_luminosity'] \
#             = comoving_to_luminosity_diff_vt_ratio(d_luminosity)
#         return aux_quantities_dataframe

#     def lnprior_ratio(self, m1_source, q, z, comoving_to_luminosity, alpha, m_min, m_max, beta_q):
#         mass_lnp = np.log(powerlaw(m1_source.values, -alpha, m_min, m_max))
#         q_min = m_min/m1_source
#         q_max = 1.
#         q_lnp = np.log(powerlaw(q.values, beta_q, q_min.values, q_max))
#         pop_lnp = mass_lnp + q_lnp + np.log(comoving_to_luminosity)
        
#         ivs_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
#         ivs_mass_lnp = ivs_mass_jacobian
#         ivs_lnp = ivs_mass_lnp 

#         return pop_lnp - ivs_lnp

# class TruncatedMassModelHyperPrior(IdentityTransformMixin, Prior):
#     standard_params = ['rate', 'alpha', 'm_min', 'm_max', 'beta_q']
#     range_dic={'rate':(5, 200),'alpha':(-4, 12), 'm_min':(2,10), 'm_max':(30, 100), 'beta_q':(-4,12)}
#     def lnprior(self, rate, alpha, m_min, m_max, beta_q):
#         log_uniform_prior = - np.log(np.prod(self.cubesize))
#         log_jeffreys_prior = - 0.5*np.log(rate)
#         return log_uniform_prior + log_jeffreys_prior


class MassPowerLawPeakToVolumetricPrior(PriorRatio):
    numerator = 'MassPowerLawPeak'
    denominator = 'VolumetricPrior'
    params = ['m1_source','q']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'comoving_to_luminosity']
    hyperparams = ['lambda_peak', 'alpha', 'm_max', 'm_mean', 'm_std', 'beta_q', 'm_min', 'delta_m']

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity']) 
        if 'comoving_to_luminosity' not in samples.keys():
            samples['comoving_to_luminosity'] = \
                comoving_to_luminosity_diff_vt_ratio(samples['d_luminosity'])

    def lnprior_ratio(self, m1_source, q, z, comoving_to_luminosity, 
                      lambda_peak, alpha, m_max, m_mean, m_std, beta_q,
                      m_min, delta_m):
        m1_source_bounds = np.array([m_min, m_max])
        a_transformed, b_transformed = (m1_source_bounds -
                                        m_mean) / m_std
        gaussian_mass_peak = np.exp(stats.truncnorm.logpdf(x=m1_source,
                                                     a=a_transformed,
                                                     b=b_transformed,
                                                     loc=m_mean,
                                                     scale=m_std))
        mass_lnp = (np.log((1-lambda_peak)*powerlaw(m1_source.values, -alpha, m_min, m_max) +
                         lambda_peak*gaussian_mass_peak) + 
                    np.log(smoothing_function(m1_source.values, m_min, delta_m)))
        q_min = m_min/m1_source
        q_max = 1.
        m2_source = q*m1_source
        q_lnp = (np.log(powerlaw(q.values, beta_q, q_min.values, q_max)) +
                 np.log(smoothing_function(m2_source.values, m_min, delta_m)))
        pop_lnp = mass_lnp + q_lnp + np.log(comoving_to_luminosity)

        ivs_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        ivs_mass_lnp = ivs_mass_jacobian
        ivs_lnp = ivs_mass_lnp 

        return pop_lnp - ivs_lnp


class MassPowerLawPeakToIASPriorRatio(PriorRatio):
    numerator = 'MassPowerLawPeak'
    denominator = 'IASPrior'
    params = ['m1_source','q']
    base_quantities = ['d_luminosity', 'q']
    derived_quantities = ['z', 'comoving_to_luminosity']
    hyperparams = ['lambda_peak', 'alpha', 'm_max', 'm_mean', 'm_std', 'beta_q', 'm_min', 'delta_m']

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity']) 
        if 'comoving_to_luminosity' not in samples.keys():
            samples['comoving_to_luminosity'] = \
                comoving_to_luminosity_diff_vt_ratio(samples['d_luminosity'])

    def lnprior_ratio(self, m1_source, q, z, comoving_to_luminosity,
                      lambda_peak, alpha, m_max, m_mean, m_std, beta_q,
                      m_min, delta_m):
        m1_source_bounds = np.array([m_min, m_max])
        a_transformed, b_transformed = (m1_source_bounds -
                                        m_mean) / m_std
        gaussian_mass_peak = np.exp(stats.truncnorm.logpdf(x=m1_source,
                                                     a=a_transformed,
                                                     b=b_transformed,
                                                     loc=m_mean,
                                                     scale=m_std))
        mass_lnp = (np.log((1-lambda_peak)*powerlaw(m1_source.values, -alpha, m_min, m_max) +
                         lambda_peak*gaussian_mass_peak) + np.log(smoothing_function(m1_source.values, m_min, delta_m)))
        q_min = m_min/m1_source
        q_max = 1.
        m2_source = q*m1_source
        q_lnp = (np.log(powerlaw(q.values, beta_q, q_min.values, q_max)) +
                 np.log(smoothing_function(m2_source.values, m_min, delta_m)))
        pop_lnp = mass_lnp + q_lnp + np.log(comoving_to_luminosity)

        ias_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        ias_mass_lnp = ias_mass_jacobian
        ias_lnp = ias_mass_lnp

        return pop_lnp - ias_lnp


class MassPowerLawPeakToLVCPriorRatio(PriorRatio):
    numerator = 'MassPowerLawPeak'
    denominator = 'LVCPrior'
    params = ['m1_source','q']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'comoving_to_luminosity']
    hyperparams = ['lambda_peak', 'alpha', 'm_max', 'm_mean', 'm_std', 'beta_q', 'm_min', 'delta_m']

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        if 'comoving_to_luminosity' not in samples.keys():
            samples['comoving_to_luminosity'] = \
                comoving_to_luminosity_diff_vt_ratio(samples['d_luminosity'])


    def lnprior_ratio(self, m1_source, q, z, comoving_to_luminosity,
                      lambda_peak, alpha, m_max, m_mean, m_std, beta_q,
                      m_min, delta_m):

        m1_source_bounds = np.array([m_min, m_max])
        gaussian_mass_peak = truncated_gaussian(m1_source,
                                                m1_source_bounds[0],
                                                m1_source_bounds[1],
                                                m_mean,
                                                m_std)
        
        mass_lnp = (np.log((1-lambda_peak)*powerlaw(m1_source.values, -alpha, m_min, m_max)
                           + lambda_peak*gaussian_mass_peak)
                    + np.log(smoothing_function(m1_source.values, m_min, delta_m)))
        
        q_min = m_min/m1_source
        q_max = 1.
        m2_source = q*m1_source
        q_lnp = (np.log(powerlaw(q.values, beta_q, q_min.values, q_max))
                 + np.log(smoothing_function(m2_source.values, m_min, delta_m)))
        
        time_dilation = - np.log(1+z)
        pop_lnp = (mass_lnp + q_lnp
                   + time_dilation)

        lvc_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        lvc_mass_lnp = lvc_mass_jacobian
        lvc_lnp = lvc_mass_lnp

        return pop_lnp - lvc_lnp

class MassPowerLawPeakNoSmoothingToLVCPriorRatio(PriorRatio):
    numerator = 'MassPowerLawPeakNoSmoothing'
    denominator = 'LVCPrior'
    params = ['m1_source','q']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z']
    hyperparams = ['lambda_peak', 'alpha', 'm_max', 'm_mean', 'm_std', 'beta', 'm_min']

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity'])

    def lnprior_ratio(self, m1_source, q, z,
                      lambda_peak, alpha, m_max, m_mean, m_std, beta,
                      m_min):

        m1_source_bounds = np.array([m_min, m_max])
        gaussian_mass_peak = truncated_gaussian(m1_source,
                                                m1_source_bounds[0],
                                                m1_source_bounds[1],
                                                m_mean,
                                                m_std)
        
        mass_lnp = np.log((1-lambda_peak)*powerlaw(m1_source.values, -alpha, m_min, m_max)
                           + lambda_peak*gaussian_mass_peak)
        
        q_min = m_min/m1_source
        q_max = 1.
        m2_source = q*m1_source
        q_lnp = (np.log(powerlaw(q.values, beta, q_min.values, q_max)))
        
        time_dilation = - np.log(1+z)
        pop_lnp = (mass_lnp + q_lnp
                   + time_dilation)

        lvc_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        lvc_mass_lnp = lvc_mass_jacobian
        lvc_lnp = lvc_mass_lnp

        return pop_lnp - lvc_lnp

class PowerLawPeakMassModelHyperPrior(IdentityTransformMixin, Prior):
    standard_params = ['rate', 'lambda_peak', 'alpha', 'm_max', 'm_mean', 
                       'm_std', 'beta_q', 'm_min', 'delta_m']
    range_dic={'rate':(5, 200),'lambda_peak':(0, 1), 'alpha':(-4, 12), 
               'm_max':(30, 100), 'm_mean':(20, 50), 'm_std':(1, 10),'beta_q':(-4, 12),
               'm_min':(2, 10), 'delta_m':(0, 10)}
    def lnprior(self, rate, lambda_peak, alpha, m_max, m_mean, m_std, beta_q, m_min, delta_m):
        log_uniform_prior = - np.log(np.prod(self.cubesize))
        log_jeffreys_prior = - 0.5*np.log(rate)
        return log_uniform_prior + log_jeffreys_prior
