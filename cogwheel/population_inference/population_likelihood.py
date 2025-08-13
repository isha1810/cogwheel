import glob
import numpy as np
from multiprocessing import Pool
from scipy.special import logsumexp

from cogwheel import utils

def logdiffexp(x, y):
    """ Evaluate log(exp(x) - exp(y)) """
    return x + np.log1p( - np.exp(y - x) )

def compute_n_eff(weights):
    """
    computes n_eff from importance weights
    """
    n_eff = np.sum(weights)**2/np.sum(weights**2)
    return n_eff

class PopulationLikelihood(utils.JSONMixin):
    def __init__(self,
                 population_to_pe_ratio,
                 ref_population_to_pe_ratio,
                 pe_to_inj_population_ratio,
                 injections_summary,
                 events_summary,
                 rate0):
        """
        Parameters
        ----------
        population_to_pe_ratio: PriorRatio
            population_model / pe_prior

        ref_population_to_pe_ratio: PriorRatio
            reference_population_model / pe_prior

        pe_to_inj_population_ratio: PriorRatio
            pe_prior / injection_prior

        pe_samples: list of pandas.DataFrame, of length `n_events`.
            Posterior samples for the events under analysis.

        list_of_evnames: list of str of length 'n_events'.
            Names of events in the same order as pe_samples are provided

        injections_summary: InjectionsSummary
            Injections Information

        events_summary: EventsSummary
            Event information

        rate0: float
            Fiducial merger rate (inverse Gpc^3 yr).

        ncores: int
            Number of cores to use in multiprocess.Pool
        """
        self.population_to_pe_ratio = population_to_pe_ratio
        self.ref_population_to_pe_ratio = ref_population_to_pe_ratio
        self.pe_to_inj_population_ratio = pe_to_inj_population_ratio

        for samples in events_summary.pe_samples_array:
            if "weights" in samples.keys():
                samples['log_weights'] = (np.log(samples['weights']) - 
                                          np.log(np.sum(samples['weights'])))
            else:
                samples['log_weights'] = -np.log(len(samples))*np.ones(len(samples))

        if "weights" in injections_summary.recovered_injections.keys():
            injections_summary.recovered_injections['log_weights'] = (
                np.log(injections_summary.recovered_injections['weights']))
        else:
            injections_summary.recovered_injections['log_weights'] = -np.log(injections_summary.n_inj)*np.ones(
                len(injections_summary.recovered_injections))

        # Add columns of derived_quantites to injections samples and PE samples
        self._add_auxiliary_quantities_to_injections_samples(
                injections_summary.recovered_injections)
        self._add_auxiliary_quantities_to_pe_samples(events_summary.pe_samples_array)

        self.n_inj = injections_summary.n_inj
        self.z = injections_summary.z
        self.t_obs = injections_summary.t_obs
        self.recovered_injections = injections_summary.recovered_injections

        self.list_of_evnames = events_summary.events
        self.pe_samples = events_summary.pe_samples_array
        self.pastro_ref = events_summary.pastros_array

        self.rate0 = rate0

        self.params = self.population_to_pe_ratio.hyperparams + ['rate']
        
        self._ln_w_denom_arr = self._compute_ln_avg_prior_ratios(
            self.ref_population_to_pe_ratio)

        self._pe_to_inj_population_ratio_lnprior_arr \
            = pe_to_inj_population_ratio.lnprior_ratio(
                **self.recovered_injections[
                    self.pe_to_inj_population_ratio.params],
                **self.recovered_injections[
                    self.pe_to_inj_population_ratio.derived_quantities])

    def lnlike(self, hyperparams_dic):
        """Log of the population likelihood."""
        shape_hyperparams = hyperparams_dic.copy()
        rate = shape_hyperparams.pop('rate')
        w_arr = self._compute_w_arr(shape_hyperparams)
        lnlike = (- rate * self._compute_vt(shape_hyperparams)
                  + np.sum(np.log(rate / self.rate0 * w_arr * self.pastro_ref
                                  + 1 - self.pastro_ref)))
        return lnlike

    def _compute_w_arr(self, shape_hyperparams):
        """
        Return
        ------
        w_arr: float array of shape (n_events,)
        """
        ln_w_numerator_arr = self._compute_ln_avg_prior_ratios(
            self.population_to_pe_ratio, **shape_hyperparams)
        w_arr = np.exp(ln_w_numerator_arr - self._ln_w_denom_arr)
        return w_arr

    def _compute_vt(self, shape_hyperparams):
        log_pop_to_inj = (self._compute_ln_prior_ratio(self.recovered_injections,
                                                self.population_to_pe_ratio,
                                                **shape_hyperparams)
                   + self._pe_to_inj_population_ratio_lnprior_arr
                  + self.recovered_injections['log_weights'])
        log_vt = (np.log(self.z) + np.log(self.t_obs)
                  + logsumexp(log_pop_to_inj ))
        log_s2 = (2 * np.log(self.t_obs) + 2 * np.log(self.z)  + np.logaddexp.reduce(
                2 * (log_pop_to_inj)))
        log_sig2 = logdiffexp(log_s2, 2.0*log_vt - np.log(self.n_inj))
        n_eff = np.exp(2 * log_vt - log_sig2)

        self.vt_n_eff = n_eff
        
        if n_eff>276:
            return np.exp(log_vt)
        else:
            return np.inf

    def _compute_vt_and_neff(self, shape_hyperparams):
        """
        Return
        ------
        vt, n_eff, err_vt
        """
        log_pop_to_inj = (self._compute_ln_prior_ratio(self.recovered_injections,
                                            self.population_to_pe_ratio,
                                            **shape_hyperparams)
                       + self._pe_to_inj_population_ratio_lnprior_arr
                      + self.recovered_injections['log_weights'])
        
        log_vt = (np.log(self.z) + np.log(self.t_obs)
                 + logsumexp(log_pop_to_inj))
        log_s2 = 2 * np.log(self.t_obs) + 2 * np.log(self.z)  + np.logaddexp.reduce(
                2 * (log_pop_to_inj))
        log_sig2 = logdiffexp(log_s2, 2.0*log_vt - np.log(self.n_inj))
        log_sig = log_sig2 / 2

        vt = np.exp(log_vt)
        n_eff = np.exp(2 * log_vt - log_sig2)
        sig = np.exp(log_sig)
        
        return vt, n_eff, sig
        
    def _compute_ln_avg_prior_ratios(self, prior_ratio, **shape_hyperparams):
        """
        Return
        ------
        float array of shape (n_events,):
            Each entry is log(mean(prior_ratio(samples))).
        """
        # n_samples = np.array([len(samples) for samples in self.pe_samples])
        # logsum_prior_ratios = np.array([
        #     logsumexp(self._compute_ln_prior_ratio(samples, prior_ratio,
        #                                            **shape_hyperparams)
        #              + samples['log_weights'])
        #     for samples in self.pe_samples])
        # return logsum_prior_ratios
        
        logsum_prior_ratios = []
        for samples in self.pe_samples:
            log_prior_ratio = (self._compute_ln_prior_ratio(samples, prior_ratio,**shape_hyperparams)
                     + samples['log_weights'])
            n_eff = compute_n_eff(np.exp(log_prior_ratio))
            if n_eff>69:
                logsum_prior_ratios.append(logsumexp(log_prior_ratio))
            else:
                logsum_prior_ratios.append(-np.inf)

        return np.asarray(logsum_prior_ratios)

    def _compute_ln_prior_ratio(
            self, samples, prior_ratio, **shape_hyperparams):
        """
        Return
        ------
        float array of shape (n_samples,)
        """
        return prior_ratio.lnprior_ratio(**samples[prior_ratio.params],
                                         **samples[prior_ratio.derived_quantities],
                                         **shape_hyperparams)

    def _add_auxiliary_quantities_to_injections_samples(
        self, injection_samples):
        """
        Return
        ------
        pandas.Dataframe with added columns corresponding to
        prior_ratio.derived_quantities
        """
        self.population_to_pe_ratio.compute_auxiliary_quantities(injection_samples)
        self.pe_to_inj_population_ratio.compute_auxiliary_quantities(injection_samples)

    def _add_auxiliary_quantities_to_pe_samples(
        self, pe_samples):
        """
        Return
        ------
        pandas.Dataframe with added columns corresponding to
        prior_ratio.derived_quantities
        """
        for samples in pe_samples:
            self.population_to_pe_ratio.compute_auxiliary_quantities(samples)
            self.ref_population_to_pe_ratio.compute_auxiliary_quantities(samples)
    
    def lnlike_and_metadata(self, par_dic):
        """
        Return log of population likelihood, and also a dict containing it
        so that it is stored with the samples.
        """
        lnl = self.lnlike(par_dic)
        return lnl, {'lnl': lnl, 'vt_n_eff': self.vt_n_eff}

    def get_blob(self, metadata):
        """
        Return dictionary of ancillary information ("blob"). This will
        be appended to the posterior samples as extra columns.
        """
        metadata.update({'vt_n_eff': self.vt_n_eff})
        return metadata

    def get_init_dict(self):
        # TODO: populate with useful information
        init_dict = {}
        return init_dict
