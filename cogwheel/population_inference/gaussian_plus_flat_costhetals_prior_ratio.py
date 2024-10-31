"""Define population models through the ratios between them."""
from scipy import stats
import numpy as np
import pandas as pd

from cogwheel.prior import (
    IdentityTransformMixin,
    UniformPriorMixin,
    Prior,
    FixedPrior,
    CombinedPrior)
from cogwheel.cosmology import (
    z_of_d_luminosity,
    comoving_to_luminosity_diff_vt_ratio)

from .base_prior_ratio import PriorRatio
from .population_likelihood import PopulationLikelihood

class GaussianPlusFlatCosThetaLSToVolumetricPriorRatio(PriorRatio):
    """
    Ratio between a (truncated) Gaussian prior on cos(theta_LS) and the
    Volumetric (flat) cos(theta_LS) prior.
    """
    numerator = 'GaussianCosThetaLS'
    denominator = 'VolumetricPrior'
    params = ['cos_theta_ls', 'm1_source']
    hyperparams = ['cos_theta_ls_mean', 'cos_theta_ls_std', 'isotropic_prior_frac']

    base_quantities = ['d_luminosity']
    derived_quantities = ['z', 'comoving_to_luminosity']

    def compute_auxiliary_quantities(self, d_luminosity):
        """
        Computes additional quantities from the d_luminosity:

        Computes redshift and comoving to luminosity

        Returns updated dataframe with new quantities
        """

        aux_quantities_dataframe = pd.DataFrame()
        aux_quantities_dataframe['z'] = z_of_d_luminosity(d_luminosity)
        aux_quantities_dataframe['comoving_to_luminosity'] \
            = comoving_to_luminosity_diff_vt_ratio(d_luminosity)

        return aux_quantities_dataframe

    def lnprior_ratio(self, cos_theta_ls, m1_source, z,
                      comoving_to_luminosity,
                      cos_theta_ls_mean, cos_theta_ls_std,
                      isotropic_prior_frac):
        """
        Return log of the ratio between a (truncated) Gaussian prior on
        cos_theta_ls and the volumetric (flat) cos_theta_ls prior.

        The Gaussian is truncated at (-1, 1).

        Parameters
        ----------
        cos_theta_ls: array of shape (n_samples,)
            Angle between total spin and orbital angular momentum for
            an event

        cos_theta_ls_mean: float
            Mean of the Gaussian (before truncation).

        cos_theta_ls_std: float
            Standard deviation of the Gaussian (before truncation).

        Return
        ------
        float array of shape (n_samples,)
        """

        beta = 1 - isotropic_prior_frac

        cos_theta_ls_bounds = np.array([-1.0, 1.0])
        a_transformed, b_transformed = (cos_theta_ls_bounds -
                                        cos_theta_ls_mean) /\
                                              cos_theta_ls_std
        gaussian_cos_theta_ls = stats.truncnorm.pdf(x=cos_theta_ls,
                                                     a=a_transformed,
                                                     b=b_transformed,
                                                     loc=cos_theta_ls_mean,
                                                     scale=cos_theta_ls_std)

        gaussian_plus_flat_cos_theta_ls = \
            isotropic_prior_frac * 0.5 + beta * gaussian_cos_theta_ls
        gaussian_plus_flat_cos_theta_ls_lnp = \
            np.log(gaussian_plus_flat_cos_theta_ls)

        volumetric_cos_theta_ls_lnp = np.log(0.5)
        mass_lnp = (np.log(300/97) -2.*np.log(m1_source)
                    + np.log(comoving_to_luminosity))

        volumetric_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        volumetric_mass_lnp = volumetric_mass_jacobian

        return (gaussian_plus_flat_cos_theta_ls_lnp -
                volumetric_cos_theta_ls_lnp
                + mass_lnp - volumetric_mass_lnp)

class UniformCosThetaLSHyperPrior(IdentityTransformMixin, Prior):
    """
    Gives the hyperprior for a gaussian + uniform distribution of
    cos(theta_LS) for a given rate, cos(theta_LS) mean, and
    cos(theta_LS) std
    """
    standard_params = [
        'rate',
        'cos_theta_ls_mean',
        'cos_theta_ls_std',
        'isotropic_prior_frac']
    range_dic={'rate':(5, 200),
               'cos_theta_ls_mean':(-1, 1),
               'cos_theta_ls_std':(0.1,2)}

    def lnprior(self, rate, cos_theta_ls_mean, cos_theta_ls_std):
        log_uniform_prior = - np.log(np.prod(self.cubesize))
        log_jeffreys_prior = - 0.5*np.log(rate)
        return log_uniform_prior + log_jeffreys_prior

class UniformCosThetaLSRateHyperPrior(IdentityTransformMixin,
                                      Prior):
    """
    Gives the hyperprior for a gaussian + uniform distribution of
    cos(theta_LS) for a uniform rate
    """
    range_dic={'rate':(5,200)}

    def lnprior(self, rate):
        log_uniform_prior = - np.log(np.prod(self.cubesize))
        log_jeffreys_prior = - 0.5*np.log(rate)
        return log_uniform_prior + log_jeffreys_prior

class UniformCosThetaLSSigmaHyperPrior(IdentityTransformMixin,
                                       UniformPriorMixin,
                                       Prior):
    """
    Gives the hyperprior for a gaussian + uniform distribution of
    cos(theta_LS) for a uniform standard deviation of the gaussian
    """
    range_dic={'cos_theta_ls_std':(0.1, 2)}

class  FixedCosThetaLSMean1Prior(FixedPrior):
    """
    Gives the hyperprior for a gaussian + uniform distribution of
    cos(theta_LS) for a fixed mean of the gaussian
    """
    standard_par_dic = {'cos_theta_ls_mean':1}

class UniformCosThetaLSMeanHyperPrior(
    IdentityTransformMixin,
    UniformPriorMixin,
    Prior):
    """
    Gives the hyperprior for a gaussian + uniform distribution of
    cos(theta_LS) for a uniform mean of the gaussian
    """
    range_dic={'cos_theta_ls_mean':(-1, 1)}

class FixedUniformPercentageFlatFracPrior(FixedPrior):
    """
    Gives the hyperprior for a gaussian + uniform distribution of
    cos(theta_LS) for a fixed proportion of the flat distribution
    defined as isotropic_prior_frac
    """
    standard_par_dic = {'isotropic_prior_frac':0}

class CombinedRateUniformMuSigmaFlatFracPriors(CombinedPrior):
    """
    Takes the individual priors for each parameter above and constructs
    a combined prior from the uniform priors
    """
    default_likelihood_class = PopulationLikelihood

    prior_classes = [UniformCosThetaLSRateHyperPrior,
                     UniformCosThetaLSSigmaHyperPrior,
                     FixedUniformPercentageFlatFracPrior,
                     UniformCosThetaLSMeanHyperPrior]

class CombinedRateUniformSigmaFlatFlatFixedMuPriors(CombinedPrior):
    """
    Takes the individual priors for each parameter above and constructs
    a combined prior from the uniform rate and sigma priors with the
    fixed mean
    """
    default_likelihood_class = PopulationLikelihood

    prior_classes = [UniformCosThetaLSRateHyperPrior,
                     UniformCosThetaLSSigmaHyperPrior,
                     FixedUniformPercentageFlatFracPrior,
                     FixedCosThetaLSMean1Prior]
