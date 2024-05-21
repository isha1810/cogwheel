"""
Compute likelihood of population model
"""
import inspect
from functools import wraps
import numpy as np
from scipy import special, stats
import matplotlib.pyplot as plt
import pandas as pd

from cogwheel import utils
from cogwheel import waveform

import cogwheel.population_inference.parametrized_prior as pp

class PopulationLikelihood(utils.JSONMixin):
    """
    Class that accesses injections summary, event posteriors and 
    population model to compute the likelihood of GW events coming
    from a particular model.
    """
    def __init__(
            self, parametrized_population, injections_summary, all_pe_samples, R0):
            # injection_population_model):
        """
        Parameters
        ----------
        parametrized_population: CombinedParametrizedPrior object
        injections_summary: distionary with keys: 'pastro_func': np.array, 
                            'recovered_injections': pandas.DataFrame, Ninj: int
        all_pe_samples: list of pandas.DataFrame s
        R0: float
        """
        self.parametrized_population = parametrized_population
        self.pastro_func = injections_summary['pastro_func']
        self.recovered_injections = self.dataframe_to_dictionary(injections_summary['recovered_injections']) 
        self.Ninj = injections_summary['Ninj']
        # self.injection_population_model = injection_population_model
        self.all_pe_samples = [self.dataframe_to_dictionary(pe_samples) for pe_samples in all_pe_samples]
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
        w_arr = np.zeros(len(self.all_pe_samples))
        lnnorm_fiducial = np.log((1200/97)*np.pi*0.14)
        for i, pe_samples in enumerate(self.all_pe_samples):
            pe_samples_trunc = pe_samples #[0:1000]
            w_arr[i] = (np.sum(np.exp(self.parametrized_population.lnprior_and_transform_samples(
                                        pe_samples_trunc, **hyperparams_dic, force_update=False) - 
                                  pe_samples_trunc['lnprior']))/
                     np.sum(np.exp(pe_samples_trunc['lnprior_fiducial'] + lnnorm_fiducial - 
                                  pe_samples_trunc['lnprior'])))
        return w_arr

    def VT(self, hyperparams_dic):
        """
        Returns population averaged sensitive volume time (Eq. 18)
        """
        VT = (1/self.Ninj) * np.sum(np.exp(self.parametrized_population.lnprior_and_transform_samples(
                                            self.recovered_injections, **hyperparams_dic, force_update=False) - 
                                    self.recovered_injections['ln_pdet_fid']))
        return VT

    def dataframe_to_dictionary(self, df):
        '''
        Convert pandas dataframe to dictionary whose keys are column
        names and values are numpy arrays
        '''
        df_dict = {col: df[col].to_numpy() for col in df.columns}
        return df_dict

    def postprocess_samples(self, samples, force_update=True):
        """
        Placeholder method that can be overriden by subclasses.
        This method will be called after sampling (e.g. marginalized
        likelihoods un-marginalize the distribution in postprocessing).
        """
        del self, samples, force_update

    def get_init_dict(self):
        init_dict = {'parametrized_population': self.parametrized_population,
                     'injections_summary_fname': 'placeholder_name',
                     'all_pe_samples_path':'placeholder_name',
                     'R0':self.R0}
                     # 'injection_population_model':self.injection_population_model}
        return init_dict
