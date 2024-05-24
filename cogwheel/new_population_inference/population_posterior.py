import json
import numpy as np

from cogwheel import gw_prior
from cogwheel import utils
from cogwheel.likelihood import (RelativeBinningLikelihood,
                                 ReferenceWaveformFinder)


class PopulationPosteriorError(Exception):
    """Error raised by the Posterior class."""

    
class PopulationPosterior(utils.JSONMixin):
    """
    Class that instantiates a prior and a likelihood and provides
    methods for sampling the posterior distribution.
    """
    def __init__(self, likelihood, hyperprior_class):
        """
        Parameters
        ----------
        injection_summary: injection summary dataframe
        population_model: Instance of `prior.Prior`
        prior: Instance of `prior.Prior`. 
        likelihood: Instance of `likelihood.PopulationModelLikelihood`
            Provides likelihood computation given a population_model object
         """
        self.likelihood = likelihood # instance of PopulationLikelihood
        self.prior = hyperprior_class # Prior class for hyperparams

        if (set(self.likelihood.population_to_pe_ratio.hyperparams) == 
                set(self.prior.sampled_params) - {'R'}):
            raise AssertionError:
                print(f"The hyperparams of the population model are not the same as prior parameters"
        
    def lnposterior(self, *args, **kwargs):
        """
        Natural logarithm of the posterior probability density in
        the space of sampled parameters (does not apply folding).
        """
        lnprior, standard_par_dic = self.prior.lnprior_and_transform(
            *args, **kwargs)
        return lnprior + self.likelihood.lnlike(standard_par_dic)
