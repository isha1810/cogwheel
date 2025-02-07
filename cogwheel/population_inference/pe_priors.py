"""
Functions to compute commonly used priors
"""
import numpy as np
from .pdfs import powerlaw

# ----------------------------------------------------------------------
# Functions that compute log priors for PE priors

def lvc_mass_lnp():
    """
    uniform in detector frame masses
    defined in params: (m1, m2)
    !!not normalized to integrate to 1!!
    """
    lnp = 0.0
    return lnp

def lvc_spin_lnp(s1x, s1y, s1z,
                 s2x, s2y, s2z, 
                 max_spin = 0.998):
    """
    isotropic in spin direction,
    uniform in spin magnitude: U(0,max_spin)
    defined in params: (s1x, s1y, s1z, s2x, s2y, s2z)
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
    cosmo => reweighted to be uniform in comoving volume
    defined in params: (m1source, m2_source, s1x, s1y, s1z, s2x, s2y, s2z)
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
    powerlaw in source frame masses, where
        alpha1 = -2.35, alpha2 = 1.0, mmin = 2., mmax = 100.
    defined in params: (m1_source, m2_source)
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
    isotropic in spin direction,
    uniform in spin magnitude: U(0,max_spin)
    defined in params: (s1x, s1y, s1z, s2x, s2y, s2z)
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