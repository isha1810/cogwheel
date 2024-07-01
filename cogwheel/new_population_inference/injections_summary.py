'''
Class that reads the injections summary from -
"utils.DATA_ROOT/injections/O3(a or b)/injection_loader/injections_summary.hdf5"
'''
import os
import h5py
import pandas as pd

# These are the directories where latest injections are for O3a and O3b
INJECTION_ROOT_DIRS = {'O3a': '/home/isha/O3a_data/injections/O3a',
                 'O3b': '/home/isha/O3a_data/injections/O3b'}
# Z = # will be same as long as same distribution and domain is used to generate injections

class InjectionsSummary:
    def __init__(n_inj, recovered_injections, pastro_ref, t_obs,
                 obs_run='O3a', sampler_name="Dynesty"):
        self.n_inj = n_inj
        self.recovered_injections = recovered_injections
        self.pastro_ref = pastro_ref
        self.t_obs = t_obs
        self.sampler_name = sampler_name
        self.z = Z

        if sampler_name == "Dynesty":
            try:
                weights = recovered_injections['weights']
                recovered_injections['importance_weights'] = weights/np.sum(weights)
            except KeyError:
                print(f"'weights' column was not found in recovered_injections")

    @classmethod
    def from_hdf5(cls, obs_run="O3a", sampler_name="Dynesty"):
        '''
        Looks for injections summary at location
        "INJECTION_ROOT_DIRS[obs_run]/injection_loader/injections_summary.hdf5"
        '''
        file_name = os.path.join(INJECTION_ROOT_DIRS[obs_run], "injection_loader",
                                "injections_summary.h5")
        recovered_injections = pd.DataFrame()
        try:
            with h5py.File(file_name, 'r') as f:
                n_inj = f['Ninj'][:][0]
                t_obs = f['TOBS'][:][0]
                pastro_ref = f['pastro'][:]
                recovered_injections_group = f['recovered_injections']
                for name, dataset in recovered_injections_group.items():
                    recovered_injections[name] = dataset[:]
        except KeyError as e:
            print(e)
            print(f"{file_name} does not contain all the information needed to create this object")

        return cls(n_inj=n_inj, recovered_injections=recovered_injections,
                  pastro_ref=pastro_ref, t_obs=t_obs, obs_run=obs_run, sampler_name=sampler_name)
    
        