# create a prior class that follows the abstract class structure of a Prior class
# add imports
from abc import ABC, abstractmethod
from cogwheel.prior import Prior

import inspect
import itertools
import pandas as pd
import numpy as np

from cogwheel import utils


class PopulationModelPrior(Prior):
    """
    Abstract prior class to define a population model
    f(theta|\lambda) - so knows how to deal with lambdas???
    """

    hyperparameter_range_dic = {}
    
    @abstractmethod
    def __init__(self, hyperparameter_range_dic, **kwargs):
        """
        Instantiate prior classes and define lambdas 
        """
        self.hyperparameter_range_dic=hyperparameter_range_dic
        super().__init__(**kwargs)

    def set_hyperparameter_range_dic(self, hyperparameter_range_dic):
        self.hyperparameter_range_dic = hyperparameter_range_dic
        return
    
    @staticmethod
    def get_init_dict():
        """
        Return dictionary with keyword arguments to reproduce the class
        instance. Subclasses should override this method if they require
        initialization parameters.
        """
        return {}

class CombinedPopulationPrior(Prior):
    """
    Make a new `Prior` subclass combining other `Prior` subclasses.

    Schematically, combine priors like [P(x), P(y|x)] → P(x, y).
    This class has a single abstract method `prior_classes` which is a
    list of `Prior` subclasses that we want to combine.
    Arguments to the `__init__` of the classes in `prior_classes` are
    passed by keyword, so it is important that those arguments have
    repeated names if and only if they are intended to have the same
    value.
    Also, the `__init__` of all classes in `prior_classes` need to
    accept `**kwargs` and forward them to `super().__init__()`.
    """
    @property
    @staticmethod
    @abstractmethod
    def prior_classes():
        """List of `Prior` subclasses with the priors to combine."""

    def __init__(self, *args, **kwargs):
        """
        Instantiate prior classes and define `range_dic`.

        The list of parameters to pass to a subclass `cls` can be found
        using `cls.init_parameters()`.
        """
        kwargs.update(dict(zip([par.name for par in self.init_parameters()],
                               args)))

        # Check for all required arguments at once:
        required = [
            par.name for par in self.init_parameters(include_optional=False)]
        if missing := [par for par in required if par not in kwargs]:
            raise TypeError(f'Missing {len(missing)} required arguments: '
                            f'{", ".join(missing)}')

        self.subpriors = [cls(**kwargs) for cls in self.prior_classes]

        self.range_dic = {}
        for subprior in self.subpriors:
            self.range_dic.update(subprior.range_dic)

        super().__init__(**kwargs)

    def __init_subclass__(cls):
        """
        Define the following attributes and methods from the combination
        of priors in `cls.prior_classes`:

            * `range_dic`
            * `hyperparameter_range_dic`
            * `standard_params`
            * `conditioned_on`
            * `periodic_params`
            * `reflective_params`
            * `folded_reflected_params`
            * `folded_shifted_params`
            * `transform`
            * `inverse_transform`
            * `lnprior_and_transform`
            * `lnprior`

        which are used to override the corresponding attributes and
        methods of the new `CombinedPopulationPrior` subclass.
        """
        super().__init_subclass__()

        cls._set_params()
        direct_params = cls.sampled_params + cls.conditioned_on
        inverse_params = cls.standard_params + cls.conditioned_on

        def transform(self, *par_vals, **par_dic):
            """
            Transform sampled parameter values to standard parameter
            values.
            Take `self.sampled_params + self.conditioned_on` parameters
            and return a dictionary with `self.standard_params`
            parameters.
            """
            par_dic.update(dict(zip(direct_params, par_vals)))
            for subprior in self.subpriors:
                input_dic = {par: par_dic[par]
                             for par in (subprior.sampled_params
                                         + subprior.conditioned_on)}
                par_dic.update(subprior.transform(**input_dic))
            return {par: par_dic[par] for par in self.standard_params}

        def inverse_transform(self, *par_vals, **par_dic):
            """
            Transform standard parameter values to sampled parameter values.
            Take `self.standard_params + self.conditioned_on` parameters and
            return a dictionary with `self.sampled_params` parameters.
            """
            par_dic.update(dict(zip(inverse_params, par_vals)))
            for subprior in self.subpriors:
                input_dic = {par: par_dic[par]
                             for par in (subprior.standard_params
                                         + subprior.conditioned_on)}
                par_dic.update(subprior.inverse_transform(**input_dic))
            return {par: par_dic[par] for par in self.sampled_params}

        def lnprior_and_transform(self, *par_vals, **par_dic):
            """
            Take sampled and conditioned-on parameters, and return a
            2-element tuple with the log of the prior and a dictionary
            with standard parameters.
            The reason for this function is that it is necessary to
            compute the transform in order to compute the prior, so if
            both are wanted it is efficient to compute them at once.
            """
            par_dic.update(dict(zip(direct_params, par_vals)))
            standard_par_dic = self.transform(**par_dic)
            par_dic.update(standard_par_dic)

            lnp = 0
            for subprior in self.subpriors:
                input_dic = {par: par_dic[par]
                             for par in (subprior.sampled_params
                                         + subprior.conditioned_on)}
                lnp += subprior.lnprior(**input_dic)
            return lnp, standard_par_dic

        def lnprior(self, *par_vals, **par_dic):
            """
            Natural logarithm of the prior probability density.
            Take `self.sampled_params + self.conditioned_on` parameters
            and return a float.
            """
            return self.lnprior_and_transform(*par_vals, **par_dic)[0]


        # Witchcraft to fix the functions' signatures:
        self_parameter = inspect.Parameter('self',
                                           inspect.Parameter.POSITIONAL_ONLY)
        direct_parameters = [self_parameter] + [
            inspect.Parameter(par, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for par in direct_params]
        inverse_parameters = [self_parameter] + [
            inspect.Parameter(par, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for par in inverse_params]
        cls._change_signature(transform, direct_parameters)
        cls._change_signature(inverse_transform, inverse_parameters)
        cls._change_signature(lnprior, direct_parameters)
        cls._change_signature(lnprior_and_transform, direct_parameters)

        cls.transform = transform
        cls.inverse_transform = inverse_transform
        cls.lnprior_and_transform = lnprior_and_transform
        cls.lnprior = lnprior

    @classmethod
    def _set_params(cls):
        """
        Set these class attributes:
            * `range_dic`
            * `hyperparameter_range_dic`
            * `standard_params`
            * `conditioned_on`
            * `periodic_params`
            * `reflective_params`
            * `folded_reflected_params`.
            * `folded_shifted_params`
        Raise `PriorError` if subpriors are incompatible.
        """
        cls.range_dic = {}
        for prior_class in cls.prior_classes:
            cls.range_dic.update(prior_class.range_dic)
            
        cls.hyperparameter_range_dic = {}
        for prior_class in cls.prior_classes:
            try:
                cls.hyperparameter_range_dic.update(prior_class.hyperparameter_range_dic)
            except AttributeError:
                continue

        for params in ('standard_params', 'conditioned_on',
                       'periodic_params', 'reflective_params',
                       'folded_reflected_params', 'folded_shifted_params',):
            setattr(cls, params, [par for prior_class in cls.prior_classes
                                  for par in getattr(prior_class, params)])

        cls.conditioned_on = list(dict.fromkeys(
            [par for par in cls.conditioned_on
             if not par in cls.standard_params]))

        # Check that the provided prior_classes can be combined:
        if len(cls.sampled_params) != len(set(cls.sampled_params)):
            raise PriorError(
                f'Priors {cls.prior_classes} cannot be combined due to '
                f'repeated sampled parameters: {cls.sampled_params}')

        if len(cls.standard_params) != len(set(cls.standard_params)):
            raise PriorError(
                f'Priors {cls.prior_classes} cannot be combined due to '
                f'repeated standard parameters: {cls.standard_params}')

        for preceding, following in itertools.combinations(
                cls.prior_classes, 2):
            for conditioned_par in preceding.conditioned_on:
                if conditioned_par in following.standard_params:
                    raise PriorError(
                        f'{following} defines {conditioned_par}, which '
                        f'{preceding} requires. {following} should come before'
                        f' {preceding}.')

    @classmethod
    def init_parameters(cls, include_optional=True):
        """
        Return list of `inspect.Parameter` objects, for the aggregated
        parameters taken by the `__init__` of `prior_classes`, without
        duplicates and sorted by parameter kind (i.e. positional
        arguments first, keyword arguments last). The `self` parameter
        is excluded.

        Parameters
        ----------
        include_optional: bool, whether to include parameters with
                          defaults in the returned list.
        """
        signatures = [inspect.signature(prior_class.__init__)
                      for prior_class in cls.prior_classes]
        all_parameters = [par for signature in signatures
                          for par in list(signature.parameters.values())[1:]]
        sorted_unique_parameters = sorted(
            dict.fromkeys(all_parameters),
            key=lambda par: (par.kind, par.default is not par.empty))

        if include_optional:
            return sorted_unique_parameters

        return [par for par in sorted_unique_parameters
                if par.default is par.empty
                and par.kind not in (par.VAR_POSITIONAL, par.VAR_KEYWORD)]

    @staticmethod
    def _change_signature(func, parameters):
        """
        Change the signature of a function to explicitize the parameters
        it takes. Use with caution.

        Parameters
        ----------
        func: function.
        parameters: sequence of `inspect.Parameter` objects.
        """
        func.__signature__ = inspect.signature(func).replace(
            parameters=parameters)

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        init_dicts = [subprior.get_init_dict() for subprior in self.subpriors]
        return utils.merge_dictionaries_safely(*init_dicts)

    @classmethod
    def get_fast_sampled_params(cls, fast_standard_params):
        """
        Return a list of parameter names that map to given "fast"
        standard parameters, useful for sampling fast-slow parameters.
        Updating fast sampling parameters is guaranteed to only change
        fast standard parameters.
        """
        return [par for prior_class in cls.prior_classes
                for par in prior_class.get_fast_sampled_params(
                    fast_standard_params)]










# # define a population model class that has a function f(theta|lambda') associated with it that defines the distribution
# # Want to interface with cogwheel and injections.py in GWIAS pipeline - easiest way to do this is to make this code work with 
# # cogwheel and get one easily loadable thing from the output of injection_loader_O3.py - think about what this would look like. 

# from abc import ABC, abstractmethod
# import numpy as np
# import math

# # class PopulationModel(ABC):
# #     '''
# #     Abstract base class to define the structure of any population model class I would like to define in the future
# #     Using abstract class instead of superclass because each population model has its own lambdas and f, etc.
# #     '''
# #     def __init__(self, injections_summary):
# #         # get injections summary a dictionary with - data frame + pastro function (think about how this will be stored)
# #         self.injections_df = injections_summary['injections_df']
# #         self.pastro_func = injections_summary['pastro_func']
# #         self.Ninj = len(injections_summary)
# #         self.VT = None
# #         self.hyperparams = []
# #         self.hyperparam_ranges = {}  
# #     def logf():
# #     def VT():

# # give this a combined prior and then 

# class PopulationModel():
    

            
# class GaussianChieff:
#     '''
#     '''
#     def __init__(self, injections_summary, Ninj):
#         self.injections_summary = injections_summary
#         self.Ninj = Ninj
#         self.VT_dict = None
#         self.hyperparams = []
#         self.hyperparam_ranges = {}
    
#     def logf(self, mean, sigma):
#         # compute the lnprior from the prior and combined prior 
#         chieff = self.injections_summary['chieff']
#         logf_gaussian = (0.5*np.log(2) - np.log(np.pi) - 2*np.log(sigma) -
#                             np.log(math.erf((mean + 1)/np.sqrt(2*sigma**2)) -
#                             math.erf((mean - 1)/np.sqrt(2*sigma**2))) -
#                             ((chieff - mean)**2 / (2*sigma**2))
#                          )
#         logf_inj = self.injections_summary['ln_pdet_fid']
#         logf_tot = logf_inj + np.log(2) + logf_gaussian
#         return logf_tot

#     def VT(self, n_samples):
#         mean_chieff_range = self.hyperparam_ranges['mean_chieff']
#         sigma_chieff_range = self.hyperparam_ranges['sigma_chieff']
#         mean_chieff_samples = np.random.uniform(mean_chieff_range[0], mean_chieff_range[1], n_samples)
#         sigma_chieff_samples = np.random.uniform(sigma_chieff_range[0], sigma_chieff_range[1], n_samples)
#         logPinj = self.injections_summary['ln_pdet_fid']
        
#         VT = np.zeros(n_samples)
#         for samp in range(n_samples):
#             logf_gaussian_chieff = self.logf(self, mean_chieff_samples[samp], sigma_chieff_samples[samp])
#             VT[samp] = (1/self.Ninj) * np.sum(np.exp(logf_gaussian_chieff - logPinj))
#         VT_dict = {self.hyperparams[0]:mean_chieff_samples, self.hyperparams[1]:sigma_chieff_samples,
#                   'VT': VT}
#         self.VT_dict = VT_dict
        
#         return VT_dict
    
#     def w_i():
#         # compute the weights using PE samples and the f defined in 
    
