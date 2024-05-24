class PopulationLikelihood:
    def _init_(self,
               population_to_pe_ratio,
               ref_population_to_pe_ratio,
               pe_to_ref_population_ratio,
               pe_samples, 
               pastro_ref, 
               # hyperparams,
               injections_summary, rate0):
        self.population_to_pe_ratio = population_to_pe_ratio
        # self.ref_population_to_pe_ratio = ref_population_to_pe_ratio
        # self.pe_to_ref_population_ratio = pe_to_ref_population_ratio

        self.ref_population_to_pe_ratio_lnprior_sum_arrs = [
            np.sum(ref_population_to_pe_ratio.lnprior_ratio(pe)) for pe in self.pe_samples]
        self.pe_to_ref_population_ratio_lnprior_arr = pe_to_ref_population_ratio.lnprior_ratio(
            self.recovered_injections)

        self.pe_samples = pe_samples
        self.pastro_ref = pastro_ref
        # self.hyperparams = hyperparams
        self.n_inj = injections_summary['Ninj']
        self.recovered_injections = injections_summary['recovered_injections']
        self.rate0 = rate0
        

    def lnlike(self, hyperparams_dic):
        lnlike = (- hyperparams_dic['R']*self._compute_vt(hyperparams_dic) + 
                np.sum(np.log((hyperparams_dic['R']/self.R0)*self._compute_w(hyperparams_dic)*self.pastro_ref)
                + (1-self.pastro_ref)))
        return lnlike

    def _compute_w(self, hyperparams_dic):
        for i, pe in enumerate(self.pe_samples):
            w[i] = (np.sum(self.population_to_pe_ratio(pe, **hyperparams_dic))/
                    self.ref_population_to_pe_ratio_lnprior_sum_arrs[i])
        return w

    def _compute_vt(self, hyperparams_dic):
        vt = (1./self.n_inj)*np.sum(self.population_to_pe_ratio.lnprior_ratio(self.pe_samples, **hyperparams_dic)*
                                  self.pe_to_ref_population_ratio_lnprior_arr)
        return vt