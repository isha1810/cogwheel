"""
Test Prior to check evidence returned
"""
import numpy as np
from cogwheel import gw_utils, cosmology
from cogwheel.cosmology import comoving_to_luminosity_diff_vt_ratio

from cogwheel.prior import Prior, CombinedPrior, IdentityTransformMixin, FixedPrior
from cogwheel.gw_prior.extrinsic import (UniformPhasePrior,
                                        IsotropicInclinationPrior,
                                        IsotropicSkyLocationPrior,
                                        UniformTimePrior,
                                        UniformPolarizationPrior,
                                        ReferenceDetectorMixin)
from cogwheel.gw_prior.combined import RegisteredPriorMixin
from cogwheel.gw_prior.miscellaneous import (ZeroTidalDeformabilityPrior,
                                             FixedReferenceFrequencyPrior)
from cogwheel.gw_prior.spin import UniformEffectiveSpinPrior, ZeroInplaneSpinsPrior
from cogwheel.population_inference.parametrized_prior import ParametrizedPrior, CombinedParametrizedPrior, HyperPrior
from scipy import interpolate

#************************* Test Cogwheel Prior ************************************
class GaussianTestPrior(IdentityTransformMixin, Prior):
    '''
    Prior to test the evidence returned by sampler
    '''
    range_dic = {'m1': (-10,10), 'm2': (-10,10)}
    standard_params=['m1', 'm2']
        
    def lnprior(self, m1, m2):
        '''
        ln prior is gaussian normalized
        '''
        Cinv = np.array([[1/(0.05)**2, 0], [0, (1/0.05)**2]])
        norm = np.sqrt((2*np.pi)**2 / np.linalg.det(Cinv))
        cube = np.array([m1,m2])
        mean = np.array([0.5, 0.5])
        lnprior_unnorm = -0.5 * (cube-mean) @ Cinv @ (cube-mean)
        lnprior_norm = -np.log(norm) + lnprior_unnorm
        
        return lnprior_norm
    
class FixedDistancePrior(FixedPrior):
    """Set distance."""
    standard_par_dic = {'d_luminosity': 1.0}

    def get_init_dict(self):
        """Dictionary with arguments to reproduce class instance."""
        return self.standard_par_dic
    
class FixedInclinationPrior(FixedPrior):
    """Set inclination."""
    standard_par_dic = {'iota': np.pi}
    
class FixedSkyLocationPrior(FixedPrior):
    """Set sky loc angles."""
    standard_par_dic = {'ra': np.pi/2, 'dec':0.0}
    
class FixedTimePrior(FixedPrior):
    """Set time."""
    standard_par_dic = {'t_geocenter': 0.0}
    
class FixedPolarizationPrior(FixedPrior):
    """Set polarization."""
    standard_par_dic = {'psi': np.pi}
    
class FixedPhasePrior(FixedPrior):
    """Set phase."""
    standard_par_dic = {'phi_ref': 0}
    
class FixedEffectiveSpinPrior(FixedPrior):
    """Set phase."""
    standard_par_dic = {'s1z': 0.0, 's2z': 0.0}
# ************************ Test Astrophysical Population **************************
class GaussianChieff(ParametrizedPrior):
    '''
    Gaussian chieff model
    '''
    standard_params = ['chieff']
    conditioned_on = []
    range_dic = {'chieff': [-1, 1]}
    hyper_params = ['chieff_mean', 'chieff_sigma']

    def __init__(self, **kwargs):
        super().__init__(self.hyper_params, **kwargs)

    def lnprior(self, chieff, chieff_mean, chieff_sigma):
        '''  
        ln prior is normalized
        '''
        lnnorm = -np.log(np.sqrt(2*np.pi)*chieff_sigma)
        lnprior_unnorm = -1/(2*chieff_sigma**2) * (chieff - chieff_mean)**2
        lnprior_norm = lnnorm + lnprior_unnorm
        
        return lnprior_norm

    def lnprior_vectorized(self, *par_vals, **par_dic):
        """
        Vectorized version of lnprior.
        """
        return self.lnprior(*par_vals, **par_dic)

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance. Subclasses should override this method if they require
        initialization parameters.
        """
        init_dict = self.range_dic
        #.update({"hyper_params": self.hyper_params})
        return init_dict


class GaussianChieffHyperPrior(HyperPrior):
    standard_params = ['R','chieff_mean', 'chieff_sigma']
    range_dic={'R':(10, 1000),'chieff_mean':(-1, 1), 'chieff_sigma':(0.1,2)}

    

# ************************ Combined Test Priors **************************
class TestPrior(RegisteredPriorMixin, CombinedPrior):
    """
    Prior class test
    """
    prior_classes = [IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,\
                     UniformPhasePrior,
                     GaussianTestPrior,
                     UniformEffectiveSpinPrior,
                     ZeroInplaneSpinsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior,
                     FixedDistancePrior]

class FixedTestPrior(RegisteredPriorMixin, CombinedPrior):
    """
    Prior class test fix everything but 2d gaussian
    """
    prior_classes = [FixedInclinationPrior,
                     FixedSkyLocationPrior,
                     FixedTimePrior,
                     FixedPolarizationPrior,
                     FixedPhasePrior,
                     GaussianTestPrior,
                     FixedEffectiveSpinPrior,
                     ZeroInplaneSpinsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior,
                     FixedDistancePrior]

class FixedTestPopulationPrior(RegisteredPriorMixin, CombinedParametrizedPrior):
    """
    Prior class test fix everything but 2d gaussian
    """
    prior_classes = [FixedInclinationPrior,
                     FixedSkyLocationPrior,
                     FixedTimePrior,
                     FixedPolarizationPrior,
                     FixedPhasePrior,
                     GaussianChieff,
                     FixedEffectiveSpinPrior,
                     ZeroInplaneSpinsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior,
                     FixedDistancePrior]
    hyper_prior_class = GaussianChieffHyperPrior


class TestingGaussianChieff(RegisteredPriorMixin, CombinedParametrizedPrior):
    """
    Prior class test fix everything but 2d gaussian
    """
    prior_classes = [IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformPhasePrior,
                     GaussianChieff,
                     ZeroInplaneSpinsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior,
                     FixedDistancePrior]
    hyper_prior_class = GaussianChieffHyperPrior

  