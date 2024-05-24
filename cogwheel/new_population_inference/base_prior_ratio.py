"""
Define the  `PriorRatio` class, which computes ratios of prior
probability densities.
These are used as intermediate products in the computation of the
population likelihood, to reweight posterior samples and injections.
"""
import inspect
from abc import ABC, abstractmethod

from cogwheel.prior import has_compatible_signature


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
        """String identifying the prior in the numerator."""
        return ''

    @classmethod
    @property
    @abstractmethod
    def denominator(cls):
        """String identifying the prior in the denominator."""
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
            raise RuntimeError(
                    f'Expected signature of `{func.__qualname__}` to accept '
                    f'{params}, got {inspect.signature(func)}.')
