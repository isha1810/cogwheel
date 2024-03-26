# create a prior class that follows the abstract class structure of a Prior class
# add imports
from abc import ABC, abstractmethod
from cogwheel.prior import Prior


class PopulationModelPrior(Prior):
    """
    Abstract prior class to define a population model
    f(theta|\lambda) - so knows how to deal with lambdas???
    """
    
    @abstractmethod
    def __init__(self, hyperparameters_range_dic, **kwargs):
        """
        Instantiate prior classes and define lambdas 
        """
        super.__init__(**kwargs)
        self.hyperparameter_range_dic = hyperparameter_range_dic
        if missing := (self.__class__.standard_par_dic.keys()
                       - self.standard_par_dic.keys()):
            raise ValueError(f'`standard_par_dic` is missing keys: {missing}')

        if extra := (self.standard_par_dic.keys()
                     - self.__class__.standard_par_dic.keys()):
            raise ValueError(f'`standard_par_dic` has extra keys: {extra}')


    def set_hyperparameter_range_dic(self, hyperparameter_range_dic):
        self.hyperparameter_range_dic = hyperparameter_range_dic
        return


# class CombinedPopulationModelPrior(Prior):
#     """
#     combine population model prior and regular prior classes
#     """
#     @property
#     @staticmethod
#     @abstractmethod
#     def prior_classes():
#         """List of `Prior` subclasses with the priors to combine."""
    
#     def __init__(self, *args, **kwargs):
#         """
#         Instantiate prior classes and define `range_dic`.

#         The list of parameters to pass to a subclass `cls` can be found
#         using `cls.init_parameters()`.
#         """
#         kwargs.update(dict(zip([par.name for par in self.init_parameters()],
#                                args)))

#         # Check for all required arguments at once:
#         required = [
#             par.name for par in self.init_parameters(include_optional=False)]
#         if missing := [par for par in required if par not in kwargs]:
#             raise TypeError(f'Missing {len(missing)} required arguments: '
#                             f'{", ".join(missing)}')

#         self.subpriors = [cls(**kwargs) for cls in self.prior_classes]

#         self.range_dic = {}
#         for subprior in self.subpriors:
#             self.range_dic.update(subprior.range_dic)

#         super().__init__(**kwargs)
        
#     def 











# # define a population model class that has a function f(theta|lambda') associated with it that defines the distribution
# # Want to interface with cogwheel and injections.py in GWIAS pipeline - easiest way to do this is to make this code work with 
# # cogwheel and get one easily loadable thing from the output of injection_loader_O3.py - think about what this would look like. 

# from abc import ABC, abstractmethod
# import numpy as np
# import math

# # class PopulationModel(ABC):
# #     '''
# #     Abstract base class to define the structure of any population model class I would like to define in the future
# #     Using abstract class instead of superclass because each population model has its own lambdas and f, etc.
# #     '''
# #     def __init__(self, injections_summary):
# #         # get injections summary a dictionary with - data frame + pastro function (think about how this will be stored)
# #         self.injections_df = injections_summary['injections_df']
# #         self.pastro_func = injections_summary['pastro_func']
# #         self.Ninj = len(injections_summary)
# #         self.VT = None
# #         self.hyperparams = []
# #         self.hyperparam_ranges = {}  
# #     def logf():
# #     def VT():

# # give this a combined prior and then 

# class PopulationModel():
    

            
# class GaussianChieff:
#     '''
#     '''
#     def __init__(self, injections_summary, Ninj):
#         self.injections_summary = injections_summary
#         self.Ninj = Ninj
#         self.VT_dict = None
#         self.hyperparams = []
#         self.hyperparam_ranges = {}
    
#     def logf(self, mean, sigma):
#         # compute the lnprior from the prior and combined prior 
#         chieff = self.injections_summary['chieff']
#         logf_gaussian = (0.5*np.log(2) - np.log(np.pi) - 2*np.log(sigma) -
#                             np.log(math.erf((mean + 1)/np.sqrt(2*sigma**2)) -
#                             math.erf((mean - 1)/np.sqrt(2*sigma**2))) -
#                             ((chieff - mean)**2 / (2*sigma**2))
#                          )
#         logf_inj = self.injections_summary['ln_pdet_fid']
#         logf_tot = logf_inj + np.log(2) + logf_gaussian
#         return logf_tot

#     def VT(self, n_samples):
#         mean_chieff_range = self.hyperparam_ranges['mean_chieff']
#         sigma_chieff_range = self.hyperparam_ranges['sigma_chieff']
#         mean_chieff_samples = np.random.uniform(mean_chieff_range[0], mean_chieff_range[1], n_samples)
#         sigma_chieff_samples = np.random.uniform(sigma_chieff_range[0], sigma_chieff_range[1], n_samples)
#         logPinj = self.injections_summary['ln_pdet_fid']
        
#         VT = np.zeros(n_samples)
#         for samp in range(n_samples):
#             logf_gaussian_chieff = self.logf(self, mean_chieff_samples[samp], sigma_chieff_samples[samp])
#             VT[samp] = (1/self.Ninj) * np.sum(np.exp(logf_gaussian_chieff - logPinj))
#         VT_dict = {self.hyperparams[0]:mean_chieff_samples, self.hyperparams[1]:sigma_chieff_samples,
#                   'VT': VT}
#         self.VT_dict = VT_dict
        
#         return VT_dict
    
#     def w_i():
#         # compute the weights using PE samples and the f defined in 
    