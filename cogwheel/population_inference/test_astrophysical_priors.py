'''
Priors and Parametrized Priors defined to use for Population Inference
'''

import numpy as np
from cogwheel import gw_utils
from cogwheel.cosmology import comoving_to_luminosity_diff_vt_ratio

from cogwheel.prior import Prior, IdentityTransformMixin
from cogwheel.population_inference.parametrized_prior import ParametrizedPrior, CombinedParametrizedPrior
from cogwheel.gw_prior.combined import RegisteredPriorMixin

from cogwheel.gw_prior.extrinsic import (UniformPhasePrior,
                                        IsotropicInclinationPrior,
                                        IsotropicSkyLocationPrior,
                                        UniformTimePrior,
                                        UniformPolarizationPrior)
from cogwheel.gw_prior.spin import ZeroInplaneSpinsPrior
from cogwheel.gw_prior.miscellaneous import (ZeroTidalDeformabilityPrior,
                                             FixedReferenceFrequencyPrior)

#import Gaussian Chieff and hyperparam class here
from cogwheel.population_inference.test_combined_parametrized_priors import GaussianChieff, GaussianChieffHyperPrior

class UniformComovingVolumePopulationPrior(IdentityTransformMixin, Prior):
    '''
    eq. 47 in Javier's Paper
    '''
    range_dic = {'d_luminosity': (10,np.inf), 
                 'comoving_to_luminosity_diff_vt_ratio':(10,np.inf)} #randomly chosen but isn't used so doesn't matter
    standard_params=['d_luminosity', 'comoving_to_luminosity_diff_vt_ratio']

    def lnprior(self, d_luminosity, comoving_to_luminosity_diff_vt_ratio):
        return np.log(4 * np.pi * d_luminosity *
                      comoving_to_luminosity_diff_vt_ratio)

    def lnprior_vectorized(self,*par_vals, **par_dic):
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
    

class PowerLawPrimarySourceFrameMassPrior(IdentityTransformMixin, Prior):
    '''
    Power law in primary source frame mass, uniform in mass ratio
    eq.44 and 45 in Javier's paper
    '''
    range_dic = {'m1_source': (3, 100),
                 'q': (1/20,1)}
    
    def __init__(self, alpha=2., **kwargs):
        super().__init__(**kwargs)
        self.alpha=alpha
        self.norm=1 # this should be such that norm*prior is normalized

    def lnprior(self, m1_source, q):
        lnprior_unnorm = -self.alpha*np.log(m1_source)
        return lnprior_unnorm + np.log(self.norm)

    def lnprior_vectorized(self,*par_vals, **par_dic):
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

#**************************mCombined Paramterized Priors*********************
class GaussianChieffPrior(RegisteredPriorMixin, CombinedParametrizedPrior):
    """
    Prior class test fix everything but 2d gaussian
    """
    prior_classes = [IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformPhasePrior,
                     UniformComovingVolumePopulationPrior,
                     PowerLawPrimarySourceFrameMassPrior,
                     GaussianChieff,
                     ZeroInplaneSpinsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]
    hyper_prior_class = GaussianChieffHyperPrior