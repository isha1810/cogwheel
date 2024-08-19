import glob
import numpy as np
from scipy.special import logsumexp

from cogwheel import utils


class PopulationLikelihood(utils.JSONMixin):
    def __init__(self,
                 population_to_pe_ratio,
                 ref_population_to_pe_ratio,
                 pe_to_inj_population_ratio,
                 pe_samples,
                 list_of_evnames,
                 injections_summary,
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

        rate0: float
            Fiducial merger rate (inverse Gpc^3 yr).
        """
        self.population_to_pe_ratio = population_to_pe_ratio
        self.ref_population_to_pe_ratio = ref_population_to_pe_ratio
        self.pe_to_inj_population_ratio = pe_to_inj_population_ratio

        for samples in pe_samples:
            if "weights" in samples.keys():
                samples['log_weights'] = (np.log(samples['weights']) - 
                                          np.log(np.sum(samples['weights'])))
            else:
                samples['log_weights'] = -np.log(len(samples))*np.ones(len(samples))

        if "weights" in injections_summary.recovered_injections.keys():
            injections_summary.recovered_injections['log_weights'] = (
                np.log(injections_summary.recovered_injections['weights']))
        else:
            injections_summary.recovered_injections['log_weights'] = -np.log(self.n_inj)*np.ones(
                len(injections_summary.recovered_injections))

        # Add columns of derived_quantites to injections samples and PE samples
        injections_summary.recovered_injections \
            = self._add_auxiliary_quantities_to_injections_samples(
                injections_summary.recovered_injections)
        pe_samples = self._add_auxiliary_quantities_to_pe_samples(
            pe_samples)

        self.pe_samples = pe_samples
        self.list_of_evnames = list_of_evnames
        self.rate0 = rate0
        self.recovered_injections = injections_summary.recovered_injections
        self.pastro_ref = self._get_pastro_array(injections_summary.pastro_ref)
        self.n_inj = injections_summary.n_inj
        self.z = injections_summary.z
        self.t_obs = injections_summary.t_obs

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
        vt = (self.z * self.t_obs) * np.sum(
            np.exp(self._compute_ln_prior_ratio(self.recovered_injections,
                                                self.population_to_pe_ratio,
                                                **shape_hyperparams)
                   + self._pe_to_inj_population_ratio_lnprior_arr
                  + self.recovered_injections['log_weights']))
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
                                                   **shape_hyperparams) 
                     + samples['log_weights'])
            for samples in self.pe_samples])

        return logsum_prior_ratios 

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
        pop_to_pe_aux_quantities \
            = self.population_to_pe_ratio.compute_auxiliary_quantities(
                **injection_samples[self.population_to_pe_ratio.base_quantities])
        pe_to_inj_aux_quantities \
            = self.pe_to_inj_population_ratio.compute_auxiliary_quantities(
                **injection_samples[self.pe_to_inj_population_ratio.base_quantities])

        injection_samples_modified = injection_samples.copy()
        utils.update_dataframe(injection_samples_modified, pop_to_pe_aux_quantities)
        utils.update_dataframe(injection_samples_modified, pe_to_inj_aux_quantities)
        
        return injection_samples_modified

    def _add_auxiliary_quantities_to_pe_samples(
        self, pe_samples):
        """
        Return
        ------
        pandas.Dataframe with added columns corresponding to
        prior_ratio.derived_quantities
        """
        pe_samples_modified = []
        for samples in pe_samples:
            pop_to_pe_aux_quantities \
                = self.population_to_pe_ratio.compute_auxiliary_quantities(
                    **samples[self.population_to_pe_ratio.base_quantities])
            ref_to_pe_aux_quantities \
                = self.ref_population_to_pe_ratio.compute_auxiliary_quantities(
                    **samples[self.ref_population_to_pe_ratio.base_quantities])
        
            # only add unique columns to dataframe
            overlapping_columns = pop_to_pe_aux_quantities.columns.intersection(
                ref_to_pe_aux_quantities.columns)
            ref_to_pe_aux_quantities_unique = ref_to_pe_aux_quantities.drop(
                columns=overlapping_columns, errors='ignore')
            unique_aux_quantities = pop_to_pe_aux_quantities.join(ref_to_pe_aux_quantities_unique, how='outer')
            samples_modified = samples.join(unique_aux_quantities, how='outer')
            pe_samples_modified.append(samples_modified)

        return pe_samples_modified

    def _get_pastro_array(self, pastro_ref_table):
        """
        Return
        ------
        Array of pastros in the same order as the pe_samples
        """
        pastro_list=[]
        for evname in self.list_of_evnames:
            try:
                pastro = pastro_ref_table[evname][0]
            except KeyError:
                print(f"pastro for event {evname} not found, using pastro=0")
                pastro = 0
            pastro_list.append(pastro)
        return np.array(pastro_list)
    
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
