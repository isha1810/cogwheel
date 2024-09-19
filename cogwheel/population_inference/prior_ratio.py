"""Define population models through the ratios between them."""
import numpy as np
import pandas as pd
from scipy import stats

from .base_prior_ratio import PriorRatio
from cogwheel.cosmology import (z_of_d_luminosity,
                                comoving_to_luminosity_diff_vt_ratio)

class GaussianChieffToIASPriorRatio(PriorRatio):
    """
    Ratio between a (truncated) Gaussian prior on chieff and the IAS
    (flat) chieff prior.
    """
    numerator = 'GaussianChieff'
    denominator = 'IASPrior'
    params = ['chieff', 'm1_source']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'comoving_to_luminosity']
    hyperparams = ['chieff_mean', 'chieff_std']

    def compute_auxiliary_quantities(self, d_luminosity):
        aux_quantities_dataframe = pd.DataFrame()
        aux_quantities_dataframe['z'] = z_of_d_luminosity(d_luminosity)
        aux_quantities_dataframe['comoving_to_luminosity'] \
            = comoving_to_luminosity_diff_vt_ratio(d_luminosity)
        
        return aux_quantities_dataframe
    
    def lnprior_ratio(self,
                      chieff,
                      m1_source,
                      z,
                      comoving_to_luminosity,
                      chieff_mean,
                      chieff_std):
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
        a_transformed, b_transformed = (chieff_bounds -
                                        chieff_mean) / chieff_std
        gaussian_chieff_lnp = stats.truncnorm.logpdf(x=chieff,
                                                     a=a_transformed,
                                                     b=b_transformed,
                                                     loc=chieff_mean,
                                                     scale=chieff_std)
        ias_chieff_lnp = np.log(0.5)
        mmin = 1.
        mass_lnp = (np.log(300/97) -2.*np.log(m1_source)
                    + np.log(comoving_to_luminosity))
        
        ias_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        ias_mass_lnp = ias_mass_jacobian
        
        return (gaussian_chieff_lnp - ias_chieff_lnp +
                mass_lnp - ias_mass_lnp)

class InjectionPriorToIASPriorRatio(PriorRatio):
    '''
    Ratio between the Injection Prior and the 
    IASPrior used for PE. This is the ratio of the f's and has no
    dimensions
    '''
    numerator = 'InjectionPrior'
    denominator = 'IASPrior'
    params = ['m1_source']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'comoving_to_luminosity']
    hyperparams = []

    def compute_auxiliary_quantities(self, d_luminosity):
        aux_quantities_dataframe = pd.DataFrame()
        aux_quantities_dataframe['z'] = z_of_d_luminosity(d_luminosity)
        aux_quantities_dataframe['comoving_to_luminosity'] \
            = comoving_to_luminosity_diff_vt_ratio(d_luminosity)
        
        return aux_quantities_dataframe
    
    def lnprior_ratio(self, m1_source, z, comoving_to_luminosity):
        alpha=2.
        mmin=1.
        injection_jacobian = (- np.log(1-(mmin/m1_source))
                        + np.log(comoving_to_luminosity))
        injection_lnp = np.log(300/97) - alpha*np.log(m1_source) +\
            injection_jacobian
        
        ias_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        ias_mass_lnp = ias_mass_jacobian
        
        return (injection_lnp - ias_mass_lnp)

class IASPriorToInjectionPriorRatio(PriorRatio):
    '''
    Ratio between the IASPrior and the Injection Prior
    (probability density).
    '''
    numerator = 'IASPrior'
    denominator = 'InjectionPrior'
    params = ['m1_source']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'comoving_to_luminosity']
    hyperparams = []

    def compute_auxiliary_quantities(self, d_luminosity):
        aux_quantities_dataframe = pd.DataFrame()
        aux_quantities_dataframe['z'] = z_of_d_luminosity(d_luminosity)
        aux_quantities_dataframe['comoving_to_luminosity'] \
            = comoving_to_luminosity_diff_vt_ratio(d_luminosity)
        
        return aux_quantities_dataframe

    def lnprior_ratio(self, m1_source, z, comoving_to_luminosity):
        alpha=2.
        mmin=1.
        injection_jacobian = (- np.log(1-(mmin/m1_source)) +
                        np.log(comoving_to_luminosity))
        injection_lnp = np.log(300/97) - alpha*np.log(m1_source) +\
            injection_jacobian
        
        ias_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        ias_mass_lnp = ias_mass_jacobian
        
        return (ias_mass_lnp - injection_lnp)

class GaussianChieffToVolumetricPrior(PriorRatio):
    """
    Return log of the ratio between a (truncated) Gaussian prior on
    chieff and the Volumetric (flat) chieff prior.

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
    numerator = 'GaussianChieff'
    denominator = 'VolumetricPrior'
    params = ['m1_source', 'chieff', 'q', 's1z', 's2z']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'comoving_to_luminosity']
    hyperparams = ['chieff_mean', 'chieff_std']

    def compute_auxiliary_quantities(self, d_luminosity):
        aux_quantities_dataframe = pd.DataFrame()
        aux_quantities_dataframe['z'] = z_of_d_luminosity(d_luminosity)
        aux_quantities_dataframe['comoving_to_luminosity'] \
            = comoving_to_luminosity_diff_vt_ratio(d_luminosity)
        
        return aux_quantities_dataframe

    def lnprior_ratio(self, m1_source, chieff, q, s1z, s2z, 
                      z, comoving_to_luminosity, chieff_mean, chieff_std):
        """
        Return log of the ratio between a (truncated)
        Gaussian prior on chieff
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
        a_transformed, b_transformed = (chieff_bounds -
                                        chieff_mean) / chieff_std
        gaussian_chieff_lnp = stats.truncnorm.logpdf(x=chieff,
                                                     a=a_transformed,
                                                     b=b_transformed,
                                                     loc=chieff_mean,
                                                     scale=chieff_std)
        mmin = 1.
        s1z_min = -1.
        s1z_max = 1.
        mass_lnp = (np.log(300/97) -2.*np.log(m1_source)
                   + np.log(comoving_to_luminosity))
        pop_lnp = gaussian_chieff_lnp + mass_lnp
        
        volumetric_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        volumetric_mass_lnp = volumetric_mass_jacobian
        volumetric_chieff_jacobian = np.log((1+q)/q) + np.log(s1z_max - s1z_min)
        volumetric_chieff_lnp = (np.log(0.75 *
                                 (1-s1z**2)) +
                                 np.log(0.75 *
                                 (1-s2z**2)) +
                                 volumetric_chieff_jacobian)
        volumetric_lnp = volumetric_mass_lnp + volumetric_chieff_lnp
        
        return pop_lnp - volumetric_lnp
    

class InjectionPriorToVolumetricPrior(PriorRatio):
    """
    ...
    """
    numerator = 'InjectionPrior'
    denominator = 'VolumetricPrior'
    params = ['m1_source', 'q', 's1z', 's2z']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'comoving_to_luminosity']
    hyperparams = []

    def compute_auxiliary_quantities(self, d_luminosity):
        aux_quantities_dataframe = pd.DataFrame()
        aux_quantities_dataframe['z'] = z_of_d_luminosity(d_luminosity)
        aux_quantities_dataframe['comoving_to_luminosity'] \
            = comoving_to_luminosity_diff_vt_ratio(d_luminosity)
        
        return aux_quantities_dataframe
    
    def lnprior_ratio(self,
                      m1_source,
                      q,
                      s1z,
                      s2z,
                      z,
                      comoving_to_luminosity):
        """
        ...
        """
        alpha=2.
        mmin=1.
        s1z_min = -1.
        s1z_max = 1.
        injection_mass_distance_jacobian = (- np.log(1-(mmin/m1_source))
                                    + np.log(comoving_to_luminosity))
        injection_mass_distance_lnp = (np.log(300/97) -
                                       alpha*np.log(m1_source) +
                                       injection_mass_distance_jacobian)
        injection_chieff_lnp = np.log(0.5)
        injection_lnp = injection_mass_distance_lnp + injection_chieff_lnp
        
        volumetric_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        volumetric_mass_lnp = volumetric_mass_jacobian
        volumetric_chieff_jacobian = np.log((1+q)/q) +\
            np.log(s1z_max - s1z_min)
        volumetric_chieff_lnp = (np.log(0.75 *
                                        (1-s1z**2)) +
                                        np.log(0.75 *
                                        (1-s2z**2)) 
                                        + volumetric_chieff_jacobian)
        volumetric_lnp = volumetric_mass_lnp + volumetric_chieff_lnp
        
        return (injection_lnp - volumetric_lnp)

class VolumetricPriorToInjectionPrior(PriorRatio):
    """
    ...
    """
    numerator = 'VolumetricPrior'
    denominator = 'InjectionPrior'
    params = ['m1_source', 'q', 's1z', 's2z']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'comoving_to_luminosity']
    hyperparams = []

    def compute_auxiliary_quantities(self, d_luminosity):
        aux_quantities_dataframe = pd.DataFrame()
        aux_quantities_dataframe['z'] = z_of_d_luminosity(d_luminosity)
        aux_quantities_dataframe['comoving_to_luminosity'] \
            = comoving_to_luminosity_diff_vt_ratio(d_luminosity)
        
        return aux_quantities_dataframe
    
    def lnprior_ratio(self,
                      m1_source,
                      q,
                      s1z,
                      s2z,
                      z,
                      comoving_to_luminosity):
        """
        ...
        """
        alpha=2.
        mmin=1.
        s1z_min = -1.
        s1z_max = 1.
        injection_mass_distance_jacobian = (- np.log(1-(mmin/m1_source)) 
                                        + np.log(comoving_to_luminosity))
        injection_mass_distance_lnp = (np.log(300/97) -
                                       alpha*np.log(m1_source) +
                                       injection_mass_distance_jacobian)
        injection_chieff_lnp = np.log(0.5)
        injection_lnp = injection_mass_distance_lnp + injection_chieff_lnp
        
        volumetric_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        volumetric_mass_lnp = volumetric_mass_jacobian
        volumetric_chieff_jacobian = np.log((1+q)/q) +\
                                     np.log(s1z_max - s1z_min)
        volumetric_chieff_lnp = (np.log(0.75 *
                                        (1-s1z**2)) +
                                        np.log(0.75 *
                                        (1-s2z**2)) +
                                        volumetric_chieff_jacobian)
        volumetric_lnp = volumetric_mass_lnp + volumetric_chieff_lnp
        
        return (volumetric_lnp - injection_lnp)

class IntrinsicVolumetricSpinPriorToInjectionPrior(PriorRatio):
    '''
    Ratio between the Volumetric Prior and the Injection Prior
    (probability density).
    '''
    numerator = 'VolumetricPrior'
    denominator = 'InjectionPrior'
    params = ['m1_source']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'comoving_to_luminosity']
    hyperparams = []

    def compute_auxiliary_quantities(self, d_luminosity):
        aux_quantities_dataframe = pd.DataFrame()
        aux_quantities_dataframe['z'] = z_of_d_luminosity(d_luminosity)
        aux_quantities_dataframe['comoving_to_luminosity'] \
            = comoving_to_luminosity_diff_vt_ratio(d_luminosity)
        
        return aux_quantities_dataframe

    def lnprior_ratio(self, m1_source, z, comoving_to_luminosity):
        alpha=2.
        mmin=1.
        injection_jacobian = (- np.log(1-(mmin/m1_source)) +
                        np.log(comoving_to_luminosity))
        injection_lnp = np.log(300/97) - alpha*np.log(m1_source) + \
            injection_jacobian

        volumetric_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        volumetric_mass_lnp = volumetric_mass_jacobian
        
        return (volumetric_mass_lnp - injection_lnp)

class InjectionPriorToIntrinsicVolumetricSpinPrior(PriorRatio):
    '''
    Ratio between the Injection Prior and the 
    Volumetric Prior used for PE. This is the ratio of the f's
    and has no dimensions
    '''
    numerator = 'InjectionPrior'
    denominator = 'VolumetricPrior'
    params = ['m1_source']
    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'comoving_to_luminosity']
    hyperparams = []

    def compute_auxiliary_quantities(self, d_luminosity):
        aux_quantities_dataframe = pd.DataFrame()
        aux_quantities_dataframe['z'] = z_of_d_luminosity(d_luminosity)
        aux_quantities_dataframe['comoving_to_luminosity'] \
            = comoving_to_luminosity_diff_vt_ratio(d_luminosity)
        
        return aux_quantities_dataframe
    
    def lnprior_ratio(self, m1_source, z, comoving_to_luminosity):
        alpha=2.
        mmin=1.
        injection_jacobian = (- np.log(1-(mmin/m1_source))
                        + np.log(comoving_to_luminosity))
        injection_lnp = np.log(300/97) - alpha*np.log(m1_source) +\
              injection_jacobian
        
        volumetric_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        volumetric_mass_lnp = volumetric_mass_jacobian
        
        return (injection_lnp - volumetric_mass_lnp)