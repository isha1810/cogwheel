"""
Functions to compute jacobian determinants
"""
import numpy as np

# ----------------------------------------------------------------------
# Mass Jacobians

def m1_m2_to_m1s_q(m1_source, z):
    """
    m1, m2 -> m1_source, q
    returns log(|det(J)|)
    """
    return np.log(m1_source) + 2*np.log(1+z)

def m1_m2_to_m1s_m2s(z):
    """
    m1, m2 -> m1_source, m2_source
    returns log(|det(J)|)
    """
    return 2*np.log(1+z)

def m1s_m2s_to_m1s_q(m1_source):
    """
    m1_source, m2_source -> m1_source, q
    returns log(|det(J)|)
    """
    return np.log(m1_source)

def m1s_cumq_to_m1s_q(m1_source, m_min):
    """
    m1_source, cum_q -> m1_source, q
    returns log(|det(J)|)
    """
    return -np.log(1 - (m_min/m1_source))

# ----------------------------------------------------------------------
# Spin Jacobians

def s1z_s2z_to_chieff_cumchidiff(q, s1z_min=-0.998, s1z_max=0.998):
    """
    s1z, s2z -> chieff, cumchidiff
    returns log(|det(J)|)
    """
    s1z_range = s1z_max-s1z_min
    return np.log(s1z_range) + np.log((1+q)/q)

# def s1z_s2z_to_chieff_chidiff():
#     """
#     s1z, s2z -> chieff, chidiff
#     returns log(|det(J)|)
#     """
#     return 