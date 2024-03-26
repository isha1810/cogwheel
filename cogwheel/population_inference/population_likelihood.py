"""
Compute likelihood of population model
"""
import sys
sys.append("../")

import inspect
from functools import wraps
import numpy as np
from scipy import special, stats
import matplotlib.pyplot as plt

from cogwheel import utils
from cogwheel import waveform

class PopulationLikelihood(utils.JSONMixin):
    """
    Class that accesses injections summary, event posteriors and 
    population model to compute the likelihood of GW events coming
    from a particular model.
    """
    def __init__(self, population_model, pastro_func, pe_samples):
        """
        Parameters
        ----------
    
        """
        self.pastro_func = pastro_func
        self.population_model = population_model
        self.pe_samples = pe_samples
        self.R0 = R0
    
    def lnlike(self, hyperparams_dic):
        """
        Returns the log likelihood of population model
        """
        lnlike = (- hyperparams_dic['R']*self.VT(hyperparams_dic) + 
                  np.sum(np.log((hyperparams_dic['R']/self.R_0)*w(hyperparams_dic)*self.p_astro_func) 
                        + (1-self.pastro_func)))
        return lnlike
        
    def w_i(self, hyperparams_dic):
        """
        Returns eq. 17
        """
        #compute later
        return w_i
        
        
    def VT(self, hyperparams_dic):
        """
        Returns eq. 
        """
        #compute later
        return VT