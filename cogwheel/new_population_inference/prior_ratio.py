"""Define population models through the ratios between them."""
from scipy import stats
import numpy as np

from base_prior_ratio import PriorRatio
from cogwheel.cosmology import z_of_d_luminosity, comoving_to_luminosity_diff_vt_ratio


class GaussianChieffToIASPriorRatio(PriorRatio):
    """
    Ratio between a (truncated) Gaussian prior on chieff and the IAS
    (flat) chieff prior.
    """
    numerator = 'GaussianChieff'
    denominator = 'IASPrior'
    params = ['chieff','m1_source', 'd_luminosity']
    hyperparams = ['chieff_mean', 'chieff_std']

    def lnprior_ratio(self, chieff, m1_source, d_luminosity, chieff_mean, chieff_std):
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
        z = z_of_d_luminosity(d_luminosity)
        mmin = 1.
        mass_lnp = -2.*np.log(m1_source) - np.log(97/300) -np.log(m1_source) - np.log(1-(mmin/m1_source))
        ias_mass_lnp =  -2*np.log(97) + 2*np.log(1+z)
        
        return gaussian_chieff_lnp - ias_chieff_lnp + mass_lnp - ias_mass_lnp

class InjectionPriortoIASPriorRatio(PriorRatio):
    '''
    Ratio between the Injection Prior and the 
    IASPrior used for PE. This is the ratio of the f's and has no
    dimensions
    '''
    numerator = 'InjectionPrior'
    denominator = 'IASPrior'
    params = ['m1_source', 'd_luminosity']
    hyperparams = []

    def lnprior_ratio(self, m1_source, d_luminosity):
        alpha=2.
        mmin=1.
        z = z_of_d_luminosity(d_luminosity)
        
        injection_jacobian = (-np.log(m1_source) - np.log(1-(mmin/m1_source)) )
                                   # + np.log(comoving_to_luminosity_diff_vt_ratio(d_luminosity)))
        injection_lnp = -alpha*np.log(m1_source) - np.log(97/300) + injection_jacobian
        
        ias_mass_jacobian = 2*np.log(1+z)
        ias_mass_lnp = -2*np.log(97) + ias_mass_jacobian
        
        return (injection_lnp - ias_mass_lnp)

class IASPriortoInjectionPriorRatio(PriorRatio):
    '''
    Ratio between the IASPrior and the Injection Prior
    (probability density).
    '''
    numerator = 'IASPrior'
    denominator = 'InjectionPrior'
    params = ['m1_source', 'd_luminosity']
    hyperparams = []

    def lnprior_ratio(self, m1_source, d_luminosity):
        alpha=2.
        mmin=1.
        z = z_of_d_luminosity(d_luminosity)
        
        injection_jacobian = (-np.log(m1_source) - np.log(1-(mmin/m1_source)) )
                                   # + np.log(comoving_to_luminosity_diff_vt_ratio(d_luminosity)))
        injection_lnp = -alpha*np.log(m1_source) - np.log(97/300) + injection_jacobian
        
        ias_mass_jacobian = 2*np.log(1+z)
        ias_mass_lnp = -2*np.log(97) + ias_mass_jacobian
        return (ias_mass_lnp - injection_lnp)
