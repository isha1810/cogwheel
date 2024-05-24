"""
Define the  `PriorRatio` class, which computes ratios of prior
probability densities.
These are used as intermediate products in the computation of the
population likelihood, to reweight posterior samples and injections.
"""
import inspect
from abc import ABC, abstractmethod

from cogwheel import utils
from cogwheel.prior import has_compatible_signature


class PriorRatioError(Exception):
    """Base class for all exceptions in this module."""


class PriorRatio(ABC):
    """
    Compute the density ratio between two prior distributions for
    parameter estimation samples.

    The prior ratio depends on event-level parameters (``params``) and
    population-level parameters (``hyperparams``).
    """
    @classmethod
    @property
    @abstractmethod
    def numerator(cls):
        """
        Object (e.g. string or tuple of strings) identifying the prior
        in the numerator.
        """
        return ''

    @classmethod
    @property
    @abstractmethod
    def denominator(cls):
        """
        Object (e.g. string or tuple of strings) identifying the prior
        in the denominator.
        """
        return ''

    @classmethod
    @property
    @abstractmethod
    def params(cls):
        """List of event-parameter names."""
        return []

    @classmethod
    @property
    @abstractmethod
    def hyperparams(cls):
        """List of population-parameter names."""
        return []

    @abstractmethod
    def lnprior_ratio(self, *args, **kwargs):
        """
        Take `*params` and `*hyperparms` and return log of the prior
        ratio.

        Implementations should be vectorized over the ``*params``
        arguments.

        Parameters
        ----------
        *params: arrays of shape (n_samples,)
            Contain value of `params` for multiple parameter estimation
            samples of a given event.

        *hyperparams: floats
            Contain a single value of `hyperparams`.

        Return
        ------
        array of shape (n_samples,)
        """
        return 0.0

    def __init_subclass__(cls):
        """
        Check that `.lnprior_ratio()` accepts `.params` and
        `.hyperparams`, raise `RuntimeError` if not.
        """
        super().__init_subclass__()

        func = cls.lnprior_ratio
        params = cls.params + cls.hyperparams
        if not has_compatible_signature(func, params):
            raise PriorRatioError(
                    f'Expected signature of `{func.__qualname__}` to accept '
                    f'{params}, got {inspect.signature(func)}.')


class CombinedPriorRatio(PriorRatio):
    """
    Make a new `PriorRatio` subclass combining other `PriorRatio`
    subclasses.

    Simply multiplies the ratios together. The `params` of the
    `prior_ratio_classes` must be disjoint, to avoid double-counting
    priors.
    """
    @classmethod
    @property
    @abstractmethod
    def prior_ratio_classes(cls):
        """List of `PriorRatio` objects."""
        return []

    @utils.ClassProperty
    def params(cls):
        """List of event-parameter names."""
        return [par
                for prior_ratio_class in cls.prior_ratio_classes
                for par in prior_ratio_class.params]

    @utils.ClassProperty
    def hyperparams(cls):
        """List of population-parameter names."""
        return [par
                for prior_ratio_class in cls.prior_ratio_classes
                for par in prior_ratio_class.hyperparams]

    def __init_subclass__(cls):
        """
        Check that the `.params` of the `prior_ratio_classes` are
        disjoint.
        """
        param_sets = [set(prior_ratio_class.params)
                      for prior_ratio_class in cls.prior_ratio_classes]
        unique_params = set().union(*param_sets)
        all_params = [par for params in param_sets for par in params]

        if len(all_params) != len(unique_params):
            raise PriorRatioError(
                '`prior_classes` should have disjoint `params`, but '
                f'{set.intersection(*param_sets)} were repeated.')

        # Witchcraft to fix the lnprior_ratio signature:
        self_parameter = inspect.Parameter('self',
                                           inspect.Parameter.POSITIONAL_ONLY)
        parameters = [self_parameter] + [
            inspect.Parameter(par, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for par in cls.params + cls.hyperparams]
        cls.lnprior_ratio.__signature__ = inspect.signature(
            cls.lnprior_ratio).replace(parameters=parameters)
        super().__init_subclass__()

    def __init__(self):
        super().__init__()
        self.prior_ratios = [prior_ratio_class()
                             for prior_ratio_class in self.prior_ratio_classes]

    def lnprior_ratio(self, *args, **kwargs):
        """
        Take `*params` and `*hyperparms` and return log of the prior
        ratio.

        Implementations should be vectorized over the ``*params``
        arguments.

        Parameters
        ----------
        *params: arrays of shape (n_samples,)
            Contain value of `params` for multiple parameter estimation
            samples of a given event.

        *hyperparams: floats
            Contain a single value of `hyperparams`.

        Return
        ------
        array of shape (n_samples,)
        """
        kwargs.update(dict(zip(self.params + self.hyperparams, args)))

        lnp_ratio = 0.0
        for prior_ratio in self.prior_ratios:
            keys = prior_ratio.params + prior_ratio.hyperparams
            kwargs_sub = {key: kwargs[key] for key in keys}
            lnp_ratio += prior_ratio.lnprior_ratio(**kwargs_sub)

        return lnp_ratio
