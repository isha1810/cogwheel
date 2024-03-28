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
    def __init__(self, population_model, injections_summary, all_pe_samples, R0):
        """
        Parameters
        ----------
    
        """
        self.population_model = population_model
        self.injection_population_model = injection_population_model
        self.pastro_func = injections_summary['pastro_func']
        self.recovered_injections = injections_summary['recovered_injections']
        self.Ninj = injections_summary['Ninj']
        self.all_pe_samples = all_pe_samples
        self.R0 = R0
    
    
    def lnlike(self, hyperparams_dic):
        """
        Returns the log likelihood of population model
        """
        lnlike = (- hyperparams_dic['R']*self.VT(hyperparams_dic) + 
                  np.sum(np.log((hyperparams_dic['R']/self.R_0)*self.w_i(hyperparams_dic)*self.p_astro_func) 
                        + (1-self.pastro_func)))
        return lnlike
        
    
    def w_i(self, hyperparams_dic):
        """
        Returns eq. 17
        """
        params = self.recovered_injections['...']
        w_i = np.zeros(len(pe_samples))
        for i, pe_samples in enumerate(self.all_pe_samples):
            w_arr[i] = (np.sum(np.exp(self.population_model.lnprior(params, hyperparams_dic) - 
                                  pe_samples['lnprior']))/
                     np.sum(np.exp(self.injection_population_model.lnprior(params, hyperparams_dic) - 
                                  pe_samples['lnprior'])))
        return w_arr
        
        
    def VT(self, hyperparams_dic):
        """
        Returns eq. 
        """
        norm_fac = ???
        params = self.recovered_injections['...']
        VT = (1/self.Ninj) * np.sum(np.exp(self.population_model.lnprior(params,hyperparams_dic) - 
                                    self.recovered_injs['lnpdet_fid']*norm_fac))
        return VT