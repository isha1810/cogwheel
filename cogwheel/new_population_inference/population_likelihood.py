import numpy as np
import copy
from scipy.special import logsumexp

class PopulationLikelihood:
    def __init__(self,
               population_to_pe_ratio,
               ref_population_to_pe_ratio,
               pe_to_ref_population_ratio,
               pe_samples, 
               pastro_ref,
               injections_summary, rate0):
        self.population_to_pe_ratio = population_to_pe_ratio
        self.ref_population_to_pe_ratio = ref_population_to_pe_ratio
        self.pe_to_ref_population_ratio = pe_to_ref_population_ratio
        self.pe_samples = pe_samples
        self.pastro_ref = pastro_ref
        self.n_inj = injections_summary['Ninj']
        self.recovered_injections = injections_summary['recovered_injections']
        self.rate0 = rate0
        
        self.params = self.population_to_pe_ratio.hyperparams + ["rate"]
        
        self.w_denom_arr = np.array([
            logsumexp(ref_population_to_pe_ratio.lnprior_ratio(
                **samples[self.ref_population_to_pe_ratio.params])) for samples in self.pe_samples])

        samples_pe_to_ref_population = self.recovered_injections[self.pe_to_ref_population_ratio.params]
        self.pe_to_ref_population_ratio_lnprior_arr = pe_to_ref_population_ratio.lnprior_ratio(
            **samples_pe_to_ref_population)

    def lnlike(self, hyperparams_dic):
        lnlike = (-hyperparams_dic['rate']*self._compute_vt(hyperparams_dic) + 
                np.sum(np.log((hyperparams_dic['rate']/self.rate0)*self._compute_w(hyperparams_dic)*self.pastro_ref)
                + (1-self.pastro_ref)))
        return lnlike

    def _compute_w(self, hyperparams_dic):
        hyperparams_model_dic = copy.deepcopy(hyperparams_dic)
        hyperparams_model_dic.pop('rate')
        w = np.zeros(len(self.pe_samples))
        for i, samples in enumerate(self.pe_samples):
            samples_pop_to_pe = samples[self.population_to_pe_ratio.params]
            w[i] = np.exp(logsumexp(self.population_to_pe_ratio.lnprior_ratio(
                **samples_pop_to_pe, **hyperparams_model_dic))
                    - self.w_denom_arr[i])
        return w

    def _compute_vt(self, hyperparams_dic):
        samples_pop_to_pe = self.recovered_injections[self.population_to_pe_ratio.params]
        hyperparams_model_dic = copy.deepcopy(hyperparams_dic)
        hyperparams_model_dic.pop('rate')
        vt = (1./self.n_inj)*np.sum(np.exp(self.population_to_pe_ratio.lnprior_ratio(
            **samples_pop_to_pe, **hyperparams_model_dic)+
                                  self.pe_to_ref_population_ratio_lnprior_arr))
        return vt