"""Define population models through the ratios between them."""
from scipy import stats
import numpy as np

from .base_prior_ratio import PriorRatio


class GaussianChieffToIASPriorRatio(PriorRatio):
    """
    Ratio between a (truncated) Gaussian prior on chieff and the IAS
    (flat) chieff prior.
    """
    numerator = 'GaussianChieff'
    denominator = 'IASPrior'
    params = ['chieff']
    hyperparams = ['chieff_mean', 'chieff_std']

    def lnprior_ratio(self, chieff, chieff_mean, chieff_std):
        """
        Return log of the ratio between a (truncated) Gaussian prior on chieff
        and the IAS (flat) chieff prior.

        The Gaussian is truncated at (-1, 1).

        Parameters
        ----------
        chieff: array of shape (n_samples,)
            Effective spin posterior samples for an event.

        chieff_mean: float
            Mean of the Gaussian (before truncation).

        chieff_std: float
            Standard deviation of the Gaussian (before truncation).

        Return
        ------
        float array of shape (n_samples,)
        """
        chieff_bounds = np.array([-1.0, 1.0])
        a_transformed, b_transformed = (chieff_bounds - chieff_mean) / chieff_std
        gaussian_chieff_lnp = stats.truncnorm.logpdf(x=chieff,
                                                     a=a_transformed,
                                                     b=b_transformed,
                                                     loc=chieff_mean,
                                                     scale=chieff_std)
        ias_chieff_lnp = np.log(0.5)
        return gaussian_chieff_lnp - ias_chieff_lnp

class FiducialModelPriortoIASPriorRatio(PriorRatio):
    '''
    Ratio between the Fiducial model Prior (here Injection Prior) and the 
    IASPrior used for PE. This is the ratio of the f's and has no
    dimensions
    '''
    numerator = 'InjectionPrior'
    denominator = 'IASPrior'
    params = ['m1', 'd_luminosity']
    hyperparams = []

    def lnprior_ratio(m1, d_luminosity):
        alpha=2.
        redshift = z_of_d_luminosity(d_luminosity)
        return (-alpha*np.log(m1) + 
                np.log(1+redshift) + 
                np.log(comoving_to_luminosity_diff_vt_ratio(d_luminosity)))

class IASPriortoInjectionPriorRatio(PriorRatio):
    '''
    Ratio between the IASPrior and the Injection Prior
    (probability density). This ratio has units of VT.
    '''
    numerator = 'IASPrior'
    denominator = 'InjectionPrior'
    params = ['m1', 'd_luminosity']
    hyperparams = []

    def lnprior_ratio(m1, d_luminosity):
        alpha=2.
        redshift = z_of_d_luminosity(d_luminosity)
        return (alpha*np.log(m1) - 
                np.log(1+redshift) - 
                np.log(comoving_to_luminosity_diff_vt_ratio(d_luminosity)))

