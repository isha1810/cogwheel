# add imports
from abc import ABC, abstractmethod
from cogwheel.prior import Prior, CombinedPrior, UniformPriorMixin, IdentityTransformMixin

import inspect
import itertools
import pandas as pd
import numpy as np
import copy

from cogwheel import utils
from cogwheel.prior import PriorError
from cogwheel.prior import FixedPrior

# TODO: rename ParametrizedPriorClass (DONE)
#     : rename CombinedParametrizedPrior(DONE)
#     : define HyperPrior class (DONE - technically not really needed, it can just inherit from regular prior class but keeping it)

# so when I create a paramterized 
#     : injections h5 file setup
#     : deal with pe samples later 
#(add columns and add lnprior)

# TODO in InjectionPrior
# write up equations with lnprior- full lnprior
# add lnprior of the inplane spins to injection lnprior
# add inlplane spin prior class

#TODO: add some way of making sure the CombinedParametrizedPrior
# can keep track of the HyperPrior associated with it
#- keep track through hyper_params and when creating a CombinedParamterizedPrior
# class, pass hyper_prior_class!!
# (for now assuming only one HyperPrior class needs to be created for
# each CombinedParametrizedPrior )
# DONE: Checking compatibility of ParametrizedPrior and HyperPrior by:
# - passing hyper_params to ParametrizedPrior
# - collecting unique hyper_params in CombinedParametrizedPrior
# - checking that the hyper_params of CombinedParametrizedPrior are the 
#sames as standard_params of associated HyperPrior

class ParametrizedPrior(Prior):
    """
    Abstract prior class to define a population model
    f(theta|\lambda) - so knows how to deal with lambdas???
    """
    standard_params=[]
    hyper_params = [] #list of hyperparameter names
    
    @abstractmethod
    def __init__(self, hyper_params, **kwargs):
        """
        Instantiate prior classes and define lambdas 
        """
        self.hyper_params=hyper_params
        super().__init__(**kwargs)

    @staticmethod
    def get_init_dict():
        """
        Return dictionary with keyword arguments to reproduce the class
        instance. Subclasses should override this method if they require
        initialization parameters.
        """
        return {}
    
    @utils.ClassProperty
    def hyper_params(self):
        """List of hyperparameter names."""
        return list(self.hyper_params)

    @utils.lru_cache()
    def transform(self, *par_vals, **par_dic):
        """
        Transform sampled parameter values to standard parameter values.
        Take `self.sampled_params + self.conditioned_on` parameters and
        return a dictionary with `self.standard_params` parameters.
        """
        par_dic.update(dict(zip(self.sampled_params + self.conditioned_on,
                                par_vals)))
        return {par: par_dic[par] for par in self.standard_params}

    inverse_transform = transform


class CombinedParametrizedPrior(Prior):
    """
    TODO: Change this description
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
        """List of `Prior` and `ParametrizedPrior` subclasses 
        with the priors to combine."""

    @property
    @staticmethod
    @abstractmethod
    def hyper_prior_class():
        """A `HyperPrior` subclass."""

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
        self.hyperprior = self.hyper_prior_class(**kwargs) #Define hyperprior attribute

        self.range_dic = {}
        for subprior in self.subpriors:
            self.range_dic.update(subprior.range_dic)

        super().__init__(**kwargs)

    def __init_subclass__(cls):
        """
        Define the following attributes and methods from the combination
        of priors in `cls.prior_classes`:

            * `range_dic`
            * `hyper_params`
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
        methods of the new `CombinedParametrizedPrior` subclass.
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

        # lnprior_and_transform_samples
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
                hyper_param_list = subprior.__dict__.get('hyper_params', [])
                input_dic = {par: par_dic[par]
                             for par in (subprior.sampled_params
                                         + subprior.conditioned_on
                                         + hyper_param_list)}
                lnp += subprior.lnprior(**input_dic)
            return lnp, standard_par_dic

        def lnprior(self, *par_vals, **par_dic):
            """
            Natural logarithm of the prior probability density.
            Take `self.sampled_params + self.conditioned_on` parameters
            and return a float.
            """
            return self.lnprior_and_transform(*par_vals, **par_dic)[0]

        def lnprior_vectorized(self, *par_vals, **par_dic):
            raise RuntimeError("Use lnprior_and_transform_samples instead")

        # This works when using dictionary for samples
        def lnprior_and_transform_samples(self, samples, **hyperparams_dic):
            """
            Natural logarithm of the prior probability density.
            Take a dataframe with `self.sampled_params + self.conditioned_on` parameters
            and return a numpy array with lnpriors.
            samples needs to be a dictionary
            """
            try:
                samples_cols = list(samples.keys())
            except AttributeError:
                print("The samples need to be a dictionary but instead ",type(samples),
                     "passed.")
                raise
                # samples_cols = list(samples.keys())
            # if force_update or \
            #         not (set(cls.standard_params) <= set(samples_cols)):
            #     direct = samples[direct_params]
            #     standard = pd.DataFrame(
            #         list(np.vectorize(self.transform)(**direct)))
            #     utils.update_dataframe(samples, standard)
            #     utils.update_dataframe(direct, standard)
            
            if not (set(cls.standard_params) <= set(samples_cols)):
                missing_params = (set(cls.standard_params) - 
                                  set(cls.standard_params).intersection(set(samples_cols)))
                raise PriorError("The samples do not contain the following keys", missing_params,
                                "that are required to compute the lnprior")
            else:
                keys_to_keep = list(set(direct_params + cls.standard_params))
                direct = {key:samples[key] for key in keys_to_keep}
                
            for key, val in hyperparams_dic.items():
                direct[key] = val

            lnp = 0
            for subprior in self.subpriors:
                if isinstance(subprior, FixedPrior):
                    input_df = direct
                else:
                    hyper_param_list = subprior.__dict__.get('hyper_params', [])
                    keys_subprior = subprior.sampled_params + subprior.conditioned_on + hyper_param_list
                    input_df = {key:direct[key] for key in keys_subprior}
                    
                lnp += subprior.lnprior_vectorized(**input_df)

            return lnp

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
        cls.lnprior_vectorized = lnprior_vectorized
        cls.lnprior_and_transform_samples = lnprior_and_transform_samples

    @classmethod
    def _set_params(cls):
        """
        Set these class attributes:
            * `range_dic`
            * `hyper_params`
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

        #set hyper_params attributes - specific to ParametrizedPrior class
        population_prior_classes = [prior_class for prior_class in cls.prior_classes
                                    if issubclass(prior_class, ParametrizedPrior)]
        cls.hyper_params = []
        for prior_class in population_prior_classes:
            cls.hyper_params.append(prior_class.hyper_params)
        unique_hyper_params= list(set([par for prior_class in population_prior_classes
                                  for par in getattr(prior_class, 'hyper_params')]))
        setattr(cls, 'hyper_params', unique_hyper_params)
        
        #set other attributes
        for params in ('standard_params','conditioned_on',
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
                    
        # Check that the HyperPrior params and the cls.hyper_params are the same
        # except for R which is the rate parameter added in the HyperPrior class
        standard_hyper_params = copy.deepcopy(cls.hyper_prior_class.standard_params)
        # First check that HyperPrior standard_params contains "R"
        try:
            standard_hyper_params.remove('R')
        except ValueError:
            raise PriorError(f'standard_params of HyperPrior class must contain rate parameter "R" '
                            f'but only contains standard_params: {cls.hyper_prior_class.standard_params}')
        # Next check that the standard_params except for rate param "R" are the same in the
        # CombinedParametrizedPrior and the HyperPrior
        if set(cls.hyper_params) != set(standard_hyper_params):
            raise PriorError(
                f'HyperPrior {cls.hyper_prior_class} with params: ' 
                f'{cls.hyper_prior_class.standard_params} not '
                f'compatible with CombinedParametrizedPrior {cls.prior_classes} '
                f' with hyper_params: {cls.hyper_params}')

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


class HyperPrior(UniformPriorMixin, IdentityTransformMixin, Prior):
    """
    Class to define the hyperparameters of the
    prior class 
    """
    standard_params = []
    range_dic={}

    @utils.ClassProperty
    def standard_params(self):
        return list(self.standard_par_dic)


# class OptimizeVectorizeMixin:
#     """
#     Adding this to ensure that:
#     1) lnprior_vectorized is required to be defined by Prior/ParametrizedPrior
#     type class so that the np.vectorize is overriden because that is very slow
#     2) So that the lnprior_and_transform_samples knows how to work with dictionaries
#     instead of dataframes
#     """
#     @abstractmethod
#     def lnprior_vectorized(self, *par_vals, **par_dic):
#         """
#         Natural logarithm of the prior probability density.
#         Take `self.sampled_params + self.conditioned_on` parameters and
#         return a float.
#         """
    
#     def __init_subclass__(cls):
#         """
#         Set ``standard_params`` to match ``sampled_params``, and check
#         that ``IdentityTransformMixin`` comes before ``Prior`` in the
#         MRO.
#         """
#         super().__init_subclass__()
#         check_inheritance_order_and_position(cls, OptimizeVectorizationMixin, Prior)

# def check_inheritance_order_and_position(subclass, base1, base2):
#     """
#     Check that class `subclass` subclasses `base1` and `base2`, in that
#     order. If it doesn't, raise `PriorError`.
#     """
#     for base in base1, base2:
#         if not issubclass(subclass, base):
#             raise PriorError(
#                 f'{subclass.__name__} must subclass {base.__name__}')

#     if subclass.mro().index(base1) != subclass.mro().index(base2)+1:
#         print(base1, subclass.mro().index(base1))
#         print(base2, subclass.mro().index(base2))
#         raise PriorError(f'Wrong inheritance order: `{subclass.__name__}` '
#                          f'must inherit from `{base1.__name__}` before '
#                          f'`{base2.__name__}` (or their subclasses).')
