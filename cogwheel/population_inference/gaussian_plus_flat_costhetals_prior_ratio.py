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
from .pdfs import powerlaw

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

class GaussianPlusFlatCosThetaLSToLVCPriorRatio(PriorRatio):
    """
    Ratio between a (truncated) Gaussian prior on cos(theta_LS) and the
    LVC cos(theta_LS) prior.
    """
    numerator = 'GaussianCosThetaLS'
    denominator = 'LVCPrior'
    params = ['cos_theta_ls', 'm1_source']
    derived_quantities = ['z','comoving_to_luminosity','q']
    
    hyperparams = ['cos_theta_ls_mean', 'cos_theta_ls_std', 'isotropic_prior_frac']

    # def compute_auxiliary_quantities(self, d_luminosity):
    #     """
    #     Computes additional quantities from the d_luminosity:

    #     Computes redshift and comoving to luminosity

    #     Returns updated dataframe with new quantities
    #     """

    #     aux_quantities_dataframe = pd.DataFrame()
    #     aux_quantities_dataframe['z'] = z_of_d_luminosity(d_luminosity)
    #     aux_quantities_dataframe['comoving_to_luminosity'] \
    #         = comoving_to_luminosity_diff_vt_ratio(d_luminosity)

    #     return aux_quantities_dataframe

    def compute_auxiliary_quantities(self, samples):
        if 'z' not in samples.keys():
            samples['z'] = z_of_d_luminosity(samples['d_luminosity'])
        if 'comoving_to_luminosity' not in samples.keys():
            samples['comoving_to_luminosity'] \
                = comoving_to_luminosity_diff_vt_ratio(samples['d_luminosity'])
        if 'q' not in samples.keys():
            samples['q'] = samples['m2'] / samples['m1']
        # add cos_theta_ls here

    def lnprior_ratio(self, cos_theta_ls, m1_source, z,
                      comoving_to_luminosity, q,
                      cos_theta_ls_mean, cos_theta_ls_std,
                      isotropic_prior_frac):
        """
        Return log of the ratio between a (truncated) Gaussian prior on
        cos_theta_ls and the LVC cos_theta_ls prior.

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

        lvc_cos_theta_ls_lnp = np.log(0.5)
        # mass_lnp = (np.log(300/97) -2.*np.log(m1_source)
                    # + np.log(comoving_to_luminosity))

        # volumetric_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        # volumetric_mass_lnp = volumetric_mass_jacobian

        # from lvc prior ratio
        # alpha1 = -2.35
        # alpha2 = 1.0
        # mmin = 2.
        # mmax = 100.
        # max_spin=0.998

        # injection_mass_jacobian = np.log(m1_source)
        # log_m1_source_norm = - np.log(np.power(mmax, alpha1+1)/(alpha1+1) - np.power(mmin, alpha1+1)/(alpha1+1))
        # log_m2_source_norm = - np.log(np.power(m1_source,alpha2+1)/(alpha2+1) - np.power(mmin,alpha2+1)/(alpha2+1))
        
        # injection_mass_lnp = (alpha1*np.log(m1_source) + alpha2*np.log(q*m1_source) 
        #                        + log_m1_source_norm + log_m2_source_norm
        #                        + injection_mass_jacobian)
        # injection_lnp = injection_mass_lnp
        
        # lvc_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        # lvc_mass_lnp = lvc_mass_jacobian
        # lvc_lnp = lvc_mass_lnp

        alpha1 = -2.35
        alpha2 = 1.0
        # mmin = 2.
        mmin = 5
        mmax = 100.
        max_spin=0.998

        injection_mass_jacobian = np.log(m1_source)
        log_m1_source_lnp = np.log(powerlaw(m1_source, alpha1, mmin, mmax))
        log_m2_source_lnp = np.log(powerlaw(m1_source*q, alpha2, mmin, m1_source.values))
        
        injection_mass_lnp = (log_m1_source_lnp + log_m2_source_lnp
                               + injection_mass_jacobian)
        injection_lnp = injection_mass_lnp
        
        lvc_mass_jacobian = 2*np.log(1+z) + np.log(m1_source)
        lvc_mass_lnp = lvc_mass_jacobian
        lvc_lnp = lvc_mass_lnp
        
        # return (injection_lnp - lvc_lnp)   

        return (gaussian_plus_flat_cos_theta_ls_lnp -
                lvc_cos_theta_ls_lnp
                + injection_lnp - lvc_lnp - np.log(1+z))
        # return(injection_lnp - lvc_lnp)# - np.log(1+z))

class GaussianPlusFlatCosThetaLSSpinMagnitudesToLVCPriorRatio(PriorRatio):
    """
    Ratio between a (truncated) Gaussian prior on cos(theta_LS) and the
    LVC cos(theta_LS) prior.

    Additionally, we add a maximum spin magnitude s_max
    """
    def __init__(self):
        super().__init__()
        self._aux_prior_ratio = GaussianPlusFlatCosThetaLSToLVCPriorRatio()

    # numerator = ''
    denominator = 'LVCPrior'
    params = ['cos_theta_ls', 'm1_source']
    derived_quantities = ['z','comoving_to_luminosity','q','s1','s2']
    
    hyperparams = ['cos_theta_ls_mean', 'cos_theta_ls_std', 'isotropic_prior_frac', 's_max']

    def compute_auxiliary_quantities(self, samples):

        self._aux_prior_ratio.compute_auxiliary_quantities(samples)
        samples['s1'] = np.sqrt(samples['s1x']**2 + samples['s1y']**2 + samples['s1z']**2)
        samples['s2'] = np.sqrt(samples['s2x']**2 + samples['s2y']**2 + samples['s2z']**2)

    def lnprior_ratio(self, cos_theta_ls, m1_source, z,
                      comoving_to_luminosity, q, s1, s2,
                      cos_theta_ls_mean, cos_theta_ls_std,
                      isotropic_prior_frac, s_max):

        lnpr = self._aux_prior_ratio.lnprior_ratio(cos_theta_ls, m1_source, z,
                      comoving_to_luminosity, q,
                      cos_theta_ls_mean, cos_theta_ls_std,
                      isotropic_prior_frac)

        assert lnpr.shape == s1.shape
        assert lnpr.shape == s2.shape
        lnpr += -2 * np.log(s_max)
        lnpr[s1 > s_max] = -np.inf
        lnpr[s2 > s_max] = -np.inf

        return lnpr

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

class FixedCosThetaLSMean1Prior(FixedPrior):
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

class ZeroIsotropicFracPrior(FixedPrior):
    """
    Gives the hyperprior for a gaussian + uniform distribution of
    cos(theta_LS) for a fixed proportion of the flat distribution
    defined as isotropic_prior_frac
    """
    standard_par_dic = {'isotropic_prior_frac':0}

class UniformIsotropicFracPrior(
    IdentityTransformMixin,
    UniformPriorMixin,
    Prior):

    range_dic = {'isotropic_prior_frac':(0,1)}

class CombinedRateUniformMuSigmaIsotropicFracPriors(CombinedPrior):

    default_likelihood_class = PopulationLikelihood

    prior_classes = [UniformCosThetaLSRateHyperPrior,
                     UniformCosThetaLSSigmaHyperPrior,
                     UniformIsotropicFracPrior,
                     UniformCosThetaLSMeanHyperPrior]

class CombinedRateUniformMuSigmaZeroIsotropicFracPriors(CombinedPrior):
    """
    Takes the individual priors for each parameter above and constructs
    a combined prior from the uniform priors
    """
    default_likelihood_class = PopulationLikelihood

    prior_classes = [UniformCosThetaLSRateHyperPrior,
                     UniformCosThetaLSSigmaHyperPrior,
                     ZeroIsotropicFracPrior,
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
                     UniformIsotropicFracPrior,
                     FixedCosThetaLSMean1Prior]

class UniformSpinPrior(
    IdentityTransformMixin,
    UniformPriorMixin,
    Prior):

    range_dic = {'s_max':(0, 1)}

class CombinedRateUniformMuSigmaIsotropicFracSpinPriors(CombinedPrior):

    default_likelihood_class = PopulationLikelihood

    prior_classes = [UniformCosThetaLSRateHyperPrior,
                     UniformCosThetaLSSigmaHyperPrior,
                     UniformSpinPrior,
                     UniformIsotropicFracPrior,
                     UniformCosThetaLSMeanHyperPrior]

class CombinedRateUniformMuSigmaSpinZeroIsotropicFracPriors(CombinedPrior):

    default_likelihood_class = PopulationLikelihood

    prior_classes = [
        UniformCosThetaLSRateHyperPrior,
        UniformCosThetaLSSigmaHyperPrior,
        UniformSpinPrior,
        ZeroIsotropicFracPrior,
        UniformCosThetaLSMeanHyperPrior
    ]

class CombinedRateUniformSigmaSpinZeroIsotropicFracPriorsOneMu(CombinedPrior):

    default_likelihood_class = PopulationLikelihood

    prior_classes = [
        UniformCosThetaLSRateHyperPrior,
        UniformCosThetaLSSigmaHyperPrior,
        UniformSpinPrior,
        ZeroIsotropicFracPrior,
        FixedCosThetaLSMean1Prior
    ]
