import numpy as np
import pandas as pd
from scipy import stats

from .base_prior_ratio import PriorRatio
from cogwheel.cosmology import z_of_d_luminosity, comoving_to_luminosity_diff_vt_ratio
from .pop_utils import normalized_powerlaw_distribution #,normalized_truncated_gaussian_distribution


class TruncatedMassModelToIntrinsicVolumetricSpinPrior(PriorRatio):
    numerator = 'TruncatedMassModel'
    denominator = 'IntrinsicVolumetricSpinPrior'
    params = ['m1_source','q']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'comoving_to_luminosity']
    hyperparams = ['alpha', 'm_min', 'm_max', 'beta_q']

    def compute_auxiliary_quantities(self, d_luminosity):
        aux_quantities_dataframe = pd.DataFrame()
        aux_quantities_dataframe['z'] = z_of_d_luminosity(d_luminosity)
        aux_quantities_dataframe['comoving_to_luminosity'] \
            = comoving_to_luminosity_diff_vt_ratio(d_luminosity)
        return aux_quantities_dataframe

    def lnprior_ratio(self, m1_source, q, z, comoving_to_luminosity, alpha, m_min, m_max, beta_q):
        mass_lnp = np.log(normalized_powerlaw_distribution(m1_source.values, -alpha, m_min, m_max))
        q_min = m_min/m1_source
        q_max = 1.
        q_lnp = np.log(normalized_powerlaw_distribution(q.values, beta_q, q_min.values, q_max))
        pop_lnp = mass_lnp + q_lnp
        
        ivs_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        ivs_mass_lnp = ivs_mass_jacobian
        ivs_lnp = ivs_mass_lnp 

        return pop_lnp - ivs_lnp


# class MassPowerLawPeakToIntrinsicVolumetricSpinPrior(PriorRatio):
#     numerator = 'MassPowerLawPeak'
#     denominator = 'IntrinsicVolumetricSpinPrior'
#     params = ['m1_source','q']
#     base_quantities = ['d_luminosity']
#     derived_quantities = ['z', 'comoving_to_luminosity']
#     hyperparams = ['lambda_peak', 'alpha', 'm_max', 'm_mean', 'm_std', 'beta_q']

#     def compute_auxiliary_quantities(self, d_luminosity):
#         aux_quantities_dataframe = pd.DataFrame()
#         aux_quantities_dataframe['z'] = z_of_d_luminosity(d_luminosity)
#         aux_quantities_dataframe['comoving_to_luminosity'] \
#             = comoving_to_luminosity_diff_vt_ratio(d_luminosity)
        
#         return aux_quantities_dataframe

#     def lnprior_ratio(self, m1_source, q, z, d_luminosity, 
#                       lambda_peak, alpha, m_max, m_mean, m_std, beta_q):
#         mass_lnp = np.log((1-lambda_peak)*normalized_powerlaw_distribution(m1_source, alpha, 3, m_max) +
#                          lambda_peak*normalized_truncated_gaussian_distribution(m1_source, m_mean, m_std, 3, m_max))
#         q_lnp = 