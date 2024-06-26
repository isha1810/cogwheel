import numpy as np
from scipy.special import logsumexp

from cogwheel import utils


class PopulationLikelihood(utils.JSONMixin):
    def __init__(self,
                 population_to_pe_ratio,
                 ref_population_to_pe_ratio,
                 pe_to_inj_population_ratio,
                 pe_samples,
                 pastro_ref,
                 injections_summary,
                 rate0,
                 injections_sampler = 'Dynesty'):
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

        pastro_ref: list of floats, of length `n_events`
            Must be in the same order as `pe_samples`.

        injections_summary: dict
            Must contain keys for ('Ninj', 'recovered_injections', Z, Tobs)
            # TODO either define an InjectionsSummary class to enforce
            this, or make `Ninj` and `recovered_injections` parameters
            to this class

        rate0: float
            Fiducial merger rate (inverse Gpc^3 yr).

        injections_sampler: str
            Name of sampler used to generate injections
            if 'Dynesty', then set importance_weights
            to samples['weights'], else importance_weights
            are all 1. Required for vt computation.
        """
        self.population_to_pe_ratio = population_to_pe_ratio
        self.ref_population_to_pe_ratio = ref_population_to_pe_ratio
        self.pe_to_inj_population_ratio = pe_to_inj_population_ratio
        self.pe_samples = pe_samples
        self.pastro_ref = pastro_ref
        self.n_inj = injections_summary['Ninj']
        self.recovered_injections = injections_summary['recovered_injections']
        self.z = injections_summary['Z']
        self.t_obs = injections_summary['T_obs']
        self.rate0 = rate0
        
        if injections_sampler == 'Dynesty':
            self.importance_weights = self.recovered_injections['weights']
        else:
            self.importance_weights = np.ones(len(self.recovered_injections))

        self.params = self.population_to_pe_ratio.hyperparams + ['rate']
        
        self._ln_w_denom_arr = self._compute_ln_avg_prior_ratios(
            self.ref_population_to_pe_ratio)

        self._pe_to_inj_population_ratio_lnprior_arr \
            = pe_to_inj_population_ratio.lnprior_ratio(
                **self.recovered_injections[
                    self.pe_to_inj_population_ratio.params])

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
        vt = (self.z * self.t_obs) / self.n_inj * np.sum(
            self.importance_weights *
                np.exp(self._compute_ln_prior_ratio(self.recovered_injections,
                                                self.population_to_pe_ratio,
                                                **shape_hyperparams)
                   + self._pe_to_inj_population_ratio_lnprior_arr))
        return vt

    def _compute_ln_avg_prior_ratios(self, prior_ratio, **shape_hyperparams):
        """
        Return
        ------
        float array of shape (n_events,):
            Each entry is log(mean(prior_ratio(samples))).
        """
        n_samples = np.array([len(samples) for samples in self.pe_samples])

        logsum_prior_ratios = np.array([
            logsumexp(self._compute_ln_prior_ratio(samples, prior_ratio,
                                                   **shape_hyperparams))
            for samples in self.pe_samples])

        return logsum_prior_ratios - np.log(n_samples)

    def _compute_ln_prior_ratio(
            self, samples, prior_ratio, **shape_hyperparams):
        """
        Return
        ------
        float array of shape (n_samples,)
        """
        return prior_ratio.lnprior_ratio(**samples[prior_ratio.params],
                                         **shape_hyperparams)

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
