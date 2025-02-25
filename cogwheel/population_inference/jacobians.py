"""
Functions to compute jacobian determinants
"""
import numpy as np

# ----------------------------------------------------------------------
# Mass Jacobians

def m1_m2_to_m1s_q(m1_source, z):
    """
    (m1, m2) -> (m1_source, q)
    returns log(|det(J)|)
    """
    return np.log(m1_source) + 2*np.log(1+z)

def m1_m2_to_m1s_m2s(z):
    """
    (m1, m2) -> (m1_source, m2_source)
    returns log(|det(J)|)
    """
    return 2*np.log(1+z)

def m1s_m2s_to_m1s_q(m1_source):
    """
    (m1_source, m2_source) -> (m1_source, q)
    returns log(|det(J)|)
    """
    return np.log(m1_source)

def m1s_cumq_to_m1s_q(m1_source, m_min):
    """
    (m1_source, cum_q) -> (m1_source, q)
    returns log(|det(J)|)
    """
    return -np.log(1 - (m_min/m1_source))

# ----------------------------------------------------------------------
# Spin Jacobians

def chieff_cumchidiff_to_s1z_s2z(chieff, q, 
                                 max_spin=0.998):
    """
    (chieff, cumchidiff) -> (s1z, s2z)
    returns log(|det(J)|)
    """
    delta_q = (1-q)/(1+q)
    lnJ = np.zeros(len(chieff))
    mask = abs(chieff)>max_spin*delta_q
    lnJ[mask] += (np.log(1-delta_q)
                    - np.log(1-abs(chieff)/max_spin))[mask]
    return lnJ


# def s1z_s2z_to_chieff_chidiff():
#     """
#     s1z, s2z -> chieff, chidiff
#     returns log(|det(J)|)
#     """
#     return 