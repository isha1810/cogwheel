"""
Functions to compute commonly used priors
"""
import numpy as np
from .pdfs import (powerlaw,
                    truncated_gaussian,
                    smoothed_uniform)
from .jacobians import chieff_cumchidiff_to_s1z_s2z

# ----------------------------------------------------------------------
# Functions that compute log priors for PE priors. #

def lvc_mass_lnp():
    """
    defined in params: (m1, m2)
    uniform in detector frame masses
    !!not normalized to integrate to 1!!
    """
    lnp = 0.0
    return lnp

def lvc_spin_lnp(s1x, s1y, s1z,
                 s2x, s2y, s2z, 
                 max_spin = 0.998):
    """
    defined in params: (s1x, s1y, s1z, s2x, s2y, s2z)
    isotropic in spin direction,
    uniform in spin magnitude: U(0,max_spin)
    """
    lnp = (- np.log(4*np.pi*(s1x**2 + s1y**2 + s1z**2)*max_spin)
                  - np.log(4*np.pi*(s2x**2 + s2y**2 + s2z**2)*max_spin))
    return lnp

# def lvc_redshift_lnp_nocosmo(z):
#     """
#     uniform in luminosity volume
#     """
#     return

# def lvc_redshift_lnp_cosmo(z):
#     """
#     uniform in comoving volume
#     """
#     return

def lvc_lnp_cosmo(s1x, s1y, s1z,
                  s2x, s2y, s2z,
                  max_spin=0.998):
    """
    defined in params: (m1source, m2_source, s1x, s1y, s1z, s2x, s2y, s2z)
    cosmo => reweighted to be uniform in comoving volume
    TODO: Add redshift prior
    """
    lnp = lvc_mass_lnp() + lvc_spin_lnp(s1x, s1y, s1z,
                                        s2x, s2y, s2z,
                                        max_spin)
    return lnp

# def lvc_lnp_nocosmo(m1_source, q,
#                      s1x, s1y, s1z, s2x, s2y, s2z,
#                      z, s1z_min=-0.998, s1z_min=0.998):
#     """
#     returns lvc_mass_lnp+lvc_spin_lnp+lvc_redshift_lnp_cosmo
#     """
#     return

# ----------------------------------------------------------------------
# Functions that compute log priors for Injection priors.

def lvc_injection_mass_lnp(m1_source, m2_source,
                           alpha1=-2.35, alpha2=1.0,
                           mmin=2.0, mmax=100.0):
    """
    defined in params: (m1_source, m2_source)
    powerlaw in source frame masses, where
        alpha1 = -2.35, alpha2 = 1.0, mmin = 2., mmax = 100.
    """
    m1_source_lnp = np.log(powerlaw(m1_source, alpha1, 
                                    mmin, mmax))
    m2_source_lnp = np.log(powerlaw(m2_source, alpha2,
                                    mmin, m1_source))
    mass_lnp = m1_source_lnp + m2_source_lnp
    return mass_lnp

def lvc_injection_spin_lnp(s1x, s1y, s1z,
                           s2x, s2y, s2z,
                           max_spin=0.998):
    """
    defined in params: (s1x, s1y, s1z, s2x, s2y, s2z)
    isotropic in spin direction,
    uniform in spin magnitude: U(0,max_spin)
    """
    return lvc_spin_lnp(s1x, s1y, s1z,
                        s2x, s2y, s2z,
                        max_spin=max_spin)

# def lvc_injection_redshift_lnp():
#     """
#     """
#     return

def lvc_injection_lnp(m1_source, m2_source,
                      s1x, s1y, s1z,
                      s2x, s2y, s2z):
    """
    defined in params: (m1source, m2_source, s1x, s1y, s1z, s2x, s2y, s2z)
    TODO: Add redshift prior
    """
    return (lvc_injection_mass_lnp(m1_source, m2_source)
            + lvc_injection_spin_lnp(s1x, s1y, s1z,
                                     s2x, s2y, s2z))

# ----------------------------------------------------------------------
# Functions that compute log priors for mass Population priors. #

def powerlaw_primary_mass_lnp(m1_source, alpha, m_min, m_max):
    """
    defined in params: (m1_source)
    
    Returns
    -------
    p(m1_source | alpha, m_min, m_max) =
                powerlaw(m1_source,-alpha,m_min, m_max)
    The above function integrates to 1 in (m_min, m_max)
    """
    primary_mass_lnp = np.log(powerlaw(m1_source.values, 
                                       -alpha,
                                       m_min,
                                       m_max))
    return primary_mass_lnp

def powerlaw_peak_primary_mass_lnp(m1_source, lambda_peak, alpha,
                          m_min, m_max, m_mean, m_std):
    """
    defined in params: (m1_source)
    
    Returns
    -------
    p(m1_source | alpha, lambda_peak,
            m_min, m_max, m_mean, m_std) =
                (1-lambda_peak)*powerlaw(m1_source,-alpha,m_min, m_max)
                    + (lambda_peak)N_t(m_mean, m_std, m_min, m_max)
    The above function integrates to 1 in (m_min, m_max)
    """
    gaussian_mass_peak = truncated_gaussian(m1_source,
                                            m_min,
                                            m_max,
                                            m_mean,
                                            m_std)
    
    primary_mass_lnp = np.log((1-lambda_peak)*powerlaw(m1_source.values, 
                                                       -alpha, m_min, m_max)
                           + lambda_peak*gaussian_mass_peak)
    return primary_mass_lnp

def powerlaw_mass_ratio_lnp(q, m1_source, beta, m_min):
    """
    defined in params: (q)
    
    Returns
    -------
    p(q | m1_source, beta, m_min) =
            powerlaw(q, beta, q_min=m_min/m1_source, q_max=1.0)
    The above function integrates to 1 in (q_min, q_max)
    """
    q_min = m_min/m1_source
    q_max = 1.
    q_lnp = np.log(powerlaw(q.values,
                            beta,
                            q_min.values,
                            q_max))
    return q_lnp

# ----------------------------------------------------------------------
# Functions that compute log priors for spin Population priors. #

def uniform_chieff_lnp(max_chieff):
    return -np.log(2*max_chieff)

def natally_spinning_prior(chieff, q, sigma_chi,
                   chieff_min=-1.0, chieff_max=1.0):
    """
    defined in params: (chieff)

    Returns
    -------
    p(chieff | q, sigma_chi) = 
                N_truncated(mu, sigma, min=chieff_min, max=chieff_max)
    where mu=0.0, sigma=sigma_chi*sqrt(1+q^2)/(1+q).
    Integrates to 1 in (chieff_min, chieff_max)
    """
    mu = 0.0
    sigma = sigma_chi*(np.sqrt(1+q**2)/(1+q))
    chieff_prior = truncated_gaussian(chieff,
                                    chieff_min,
                                    chieff_max,
                                    mu,
                                    sigma.values)
    return chieff_prior

def tidally_locked_secondary_spin_prior(chieff, q, sigma_chi,
                                     chieff_min=-1.0, chieff_max=1.0):
    """
    defined in params: (chieff)

    Returns
    -------
    p(chieff | q, sigma_chi) = 
                N_truncated(mu, sigma, min=chieff_min, max=chieff_max)
    where mu=q/(1+q), sigma=sigma_chi/(1+q).
    Integrates to 1 in (chieff_min, chieff_max).
    """
    mu = q/(1+q)
    sigma = sigma_chi/(1+q)
    chieff_prior = truncated_gaussian(chieff,
                                    chieff_min,
                                    chieff_max,
                                    mu.values,
                                    sigma.values)
    return chieff_prior

def tidally_torqued_secondary_spin_prior(chieff, q, sigma_chi):
    """
    defined in params: (chieff)

    Returns
    -------
    p(chieff | q, sigma_chi) = 
                    U_smoothed(0, q/(1+q), sigma_chi)
    """
    chieff_prior = smoothed_uniform(chieff, 
                                  0.0,
                                  q/(1+q),
                                  sigma_chi)
    return chieff_prior

def uniform_chieff_cartesian_spins_lnp(
                            s1x, s1y, s1z, s2x, s2y, s2z, q,
                            max_spin=0.998):
    """
    defined in params: (s1x, s1y, s1z, s2x, s2y, s2z)
    
    Returns
    -------
    p(s1x, s1y, s1z, s2x, s2y, s2z) =
                    1/(pi^2*(max_spin^2 - s1z^2)) * 1/(pi^2*(max_spin^2 - s2z^2))
                     * 1/(4*(1+q)*max_spin^2) *
                         1                       for |chieff|<= max_spin*(1-q)/(1+q)
                         1/(1-|chieff|/max_spin) for |chieff|> max_spin*(1-q)/(1+q)
    """
    chieff = (s1z + q*s2z)/(1+q)
    spin_prior = (-np.log(np.pi*(max_spin**2-s1z**2))
                    -np.log(np.pi*(max_spin**2-s2z**2))
                    +chieff_cumchidiff_to_s1z_s2z(chieff, q, 
                                 max_spin=max_spin)
                    -np.log(4*(1+q)*max_spin**2))
    return spin_prior
# ----------------------------------------------------------------------
# Functions that compute log prior ratios for spin priors.
