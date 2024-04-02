"""
Define the Posterior class.
Can run as a script to make a Posterior instance from scratch and find
the maximum likelihood solution on the full parameter space.
"""

import argparse
import inspect
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
    def __init__(self, injection_summary, population_model,
                 hyperparameter_prior, likelihood):
        """
        Parameters
        ----------
        injection_summary: injection summary dataframe
        population_model: Instance of `prior.Prior`
        prior: Instance of `prior.Prior`. 
        likelihood: Instance of `likelihood.PopulationModelLikelihood`
            Provides likelihood computation given a population_model object
         """
        self.injection_summary = injection_summary #read from hdf5 file
        self.population_model = population_model #instance of CombinedPopulationPrior 
        self.likelihood = likelihood # instance of PopulationLikelihood
        
        self.prior = hyperparameter_prior # hyperparameter prior object, instance of CombinedPrior
    
        
    def lnposterior(self, *args, **kwargs):
        """
        Natural logarithm of the posterior probability density in
        the space of sampled parameters (does not apply folding).
        """
        lnprior, standard_par_dic = self.prior.lnprior_and_transform(
            *args, **kwargs)
        return lnprior + self.likelihood.lnlike(standard_par_dic)

    def create_uniform_hyperparam_prior_class(hyperparam_range_dic):
        class UniformHyperparamPrior(UniformPriorMixin, Prior):
            def __init__(self):
                self.range_dic = hyperparam_range_dic

        hyperparam_prior_class_list = []
        CombinedHyperparamPrior = 
    