# define a population model class that has a function f(theta|lambda') associated with it that defines the distribution
# Want to interface with cogwheel and injections.py in GWIAS pipeline - easiest way to do this is to make this code work with 
# cogwheel and get one easily loadable thing from the output of injection_loader_O3.py - think about what this would look like. 

from abc import ABC, abstractmethod
import numpy as np
import math

# class PopulationModel(ABC):
#     '''
#     Abstract base class to define the structure of any population model class I would like to define in the future
#     Using abstract class instead of superclass because each population model has its own lambdas and f, etc.
#     '''
#     def __init__(self, injections_summary):
#         # get injections summary a dictionary with - data frame + pastro function (think about how this will be stored)
#         self.injections_df = injections_summary['injections_df']
#         self.pastro_func = injections_summary['pastro_func']
#         self.Ninj = len(injections_summary)
#         self.VT = None
#         self.hyperparams = []
#         self.hyperparam_ranges = {}  
#     def logf():
#     def VT():
        
            
class GaussianChieff:
    '''
    '''
    def __init__(self, injections_summary, Ninj):
        self.injections_summary = injections_summary
        self.Ninj = Ninj
        self.VT_dict = None
        self.hyperparams = []
        self.hyperparam_ranges = {}
    
    @staticmethod
    def logf(self, mean, sigma):
        chieff = self.injections_summary['chieff']
        logf_gaussian = (0.5*np.log(2) - np.log(np.pi) - 2*np.log(sigma) -
                            np.log(math.erf((mean + 1)/np.sqrt(2*sigma**2)) -
                            math.erf((mean - 1)/np.sqrt(2*sigma**2))) -
                            ((chieff - mean)**2 / (2*sigma**2))
                         )
        logf_inj = self.injections_summary['ln_pdet_fid']
        logf_tot = logf_inj + np.log(2) + logf_gaussian
        return logf_tot

    @staticmethod
    def VT(self, n_samples):
        mean_chieff_range = self.hyperparam_ranges['mean_chieff']
        sigma_chieff_range = self.hyperparam_ranges['sigma_chieff']
        mean_chieff_samples = np.random.uniform(mean_chieff_range[0], mean_chieff_range[1], n_samples)
        sigma_chieff_samples = np.random.uniform(sigma_chieff_range[0], sigma_chieff_range[1], n_samples)
        logPinj = self.injections_summary['ln_pdet_fid']
        
        VT = np.zeros(n_samples)
        for samp in range(n_samples):
            logf_gaussian_chieff = self.logf(self, mean_chieff_samples[samp], sigma_chieff_samples[samp])
            VT[samp] = (1/self.Ninj) * np.sum(np.exp(logf_gaussian_chieff - logPinj))
        VT_dict = {self.hyperparams[0]:mean_chieff_samples, self.hyperparams[1]:sigma_chieff_samples,
                  'VT': VT}
        self.VT_dict = VT_dict
        
        return VT_dict
    