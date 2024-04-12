"""
Compute likelihood of population model
"""
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
    def __init__(self, parametrized_population, injections_summary, all_pe_samples, R0, injection_population_model):
        """
        Parameters
        ----------
    
        """
        self.parametrized_population = parametrized_population
        self.pastro_func = injections_summary['pastro_func']
        self.recovered_injections = injections_summary['recovered_injections']
        self.Ninj = injections_summary['Ninj']
        self.injection_population_model = injection_population_model
        self.all_pe_samples = all_pe_samples
        self.R0 = R0
    
    
    def lnlike(self, hyperparams_dic):
        """
        Returns the log likelihood of population model (Eq. 16)
        """
        lnlike = (- hyperparams_dic['R']*self.VT(hyperparams_dic) + 
                  np.sum(np.log((hyperparams_dic['R']/self.R0)*self.w_i(hyperparams_dic)*self.pastro_func) 
                        + (1-self.pastro_func)))
        return lnlike
        
    
    def w_i(self, hyperparams_dic):
        """
        Returns Eq. 17
        """
        pe_param_keys = np.array(self.parametrized_population.sampled_params) # lnprior function takes sampled params as input
        inj_param_keys = np.array(self.injection_population_model.sampled_params)
        
        #params_df = self.recovered_injections[param_keys]
        #all_params_dict = {col: params_df[col].to_numpy() for col in params_df.columns}
        #all_params_dict.update(hyperparams_dic)
        
        #vectorized_population_lnprior = np.vectorize(self.parametrized_population.lnprior)
        #vectorized_injection_population_lnprior = np.vectorize(self.injection_population_model.lnprior)

        vectorized_population_lnprior = self.parametrized_population.lnprior
        vectorized_injection_population_lnprior = self.injection_population_model.lnprior
        
        w_arr = np.zeros(len(self.all_pe_samples))
        for i, pe_samples in enumerate(self.all_pe_samples):
            pe_model_params_df = pe_samples[pe_param_keys]
            model_params_dict = {col: pe_model_params_df[col].to_numpy() for col in pe_model_params_df.columns}
            model_params_dict.update(hyperparams_dic)
            
            pe_inj_params_df = pe_samples[inj_param_keys]
            inj_params_dict = {col: pe_inj_params_df[col].to_numpy() for col in pe_inj_params_df.columns}
            
            w_arr[i] = (np.sum(np.exp(vectorized_population_lnprior(**model_params_dict) - 
                                  pe_samples['lnl']))/
                     np.sum(np.exp(vectorized_injection_population_lnprior(**inj_params_dict) - 
                                  pe_samples['lnl'])))
        return w_arr
        
        
    def VT(self, hyperparams_dic):
        """
        Returns population averaged sensitive volume time (Eq. 18)
        """
        norm_fac = 1 #CHANGE THIS TO CORRECT VALUE
        param_keys = np.array(self.parametrized_population.sampled_params)
        params_df = self.recovered_injections[param_keys]
        all_params_dict = {col: params_df[col].to_numpy() for col in params_df.columns}
        all_params_dict.update(hyperparams_dic)

        #vectorized_population_lnprior = np.vectorize(self.parametrized_population.lnprior)
        #vectorized_injection_population_lnprior = np.vectorize(self.injection_population_model.lnprior)

        vectorized_population_lnprior = self.parametrized_population.lnprior
        
        VT = (1/self.Ninj) * np.sum(np.exp(vectorized_population_lnprior(**all_params_dict) - 
                                    self.recovered_injections['ln_pdet_fid']*norm_fac))
        return VT

    def get_init_dict(self):
        init_dict = {'parametrized_population': self.parametrized_population,
                     'injections_summary_fname': 'placeholder_name',
                     'all_pe_samples_path':'placeholder_name',
                     'R0':self.R0,
                     'injection_population_model':self.injection_population_model}
        return init_dict
                     
        
