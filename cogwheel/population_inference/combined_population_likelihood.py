'''
returns the lnlike of multiple runs
'''
import numpy as np

from cogwheel import utils

def logdiffexp(x, y):
    ''' Evaluate log(exp(x) - exp(y)) '''
    return x + np.log1p( - np.exp(y - x) )

class CombinedPopulationLikelihood(utils.JSONMixin):
    """
    combines multiple runs 
    """
    def __init__(self, population_likelihood_list):
        """
        Parameters
        ----------
        population_likelihood_list: list of PopulationLikelihood objects
            
        """
        # check to make sure the population_to_pe have the same hyperparams
        hyperparams_list = population_likelihood_list.hyperparams
        for population_likelihood_object in population_likelihood_list[1:]:
            assert set(hyperparams_list) == \
                    set(population_likelihood_object.hyperparams),\
                    "The PopulationLikelihood objects passed must have the same 'params'."
        #then do
        self.population_likelihood_list = population_likelihood_list
        self.params = self.population_to_pe_ratio.hyperparams + ['rate']

    def lnlike(self, hyperparams_dic):
        """Log of the population likelihood."""
        lnlike = np.sum([pop_like.lnlike(hyperparams_dic) 
                         for pop_like in self.population_likelihood_list])
        return lnlike
    
    def lnlike_and_metadata(self, par_dic):
        """
        Return log of population likelihood, and also a dict containing it
        so that it is stored with the samples.
        """
        lnl = self.lnlike(par_dic)
        return lnl, {'lnl': lnl}

    def get_blob(self, metadata):
        """
        Return dictionary of ancillary information ("blob"). This will
        be appended to the posterior samples as extra columns.
        """
        return metadata

    def get_init_dict(self):
        # TODO: populate with useful information
        init_dict = {}
        return init_dict
