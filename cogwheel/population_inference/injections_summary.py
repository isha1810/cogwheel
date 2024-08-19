'''
Class that reads the injections summary from -
"utils.DATA_ROOT/injections/O3(a or b)/injection_loader/injections_summary.hdf5"
'''
import h5py
import numpy as np
import os
import pandas as pd

# These are the directories where latest injections are for O3a and O3b
# INJECTION_ROOT_DIRS = {'O3a': '/home/isha/O3a_data/injections/O3a',
#                  'O3b': '/home/isha/O3a_data/injections/O3b'}
SUMMARY_FILE_PATHS = {'O3a': os.path.join('data',
                                "injections_summary_pastro_added.hdf5")}
                     # 'O3b': os.path.join(INJECTION_ROOT_DIRS['O3b'], "injection_loader",
                     #            "injections_summary.hdf5")}
Z = 2.15 # Gpc^3 # same for O3a, O3b

class InjectionsSummary:
    def __init__(self, n_inj, t_obs, pastro_ref, recovered_injections,
                 obs_run='O3a', ifar_threshold=0.5):
        """
        Parameters
        ----------
        n_inj: int
            Number of waveforms injected

        t_obs: float
            duration of observing run (in yrs)

        pastro_ref: pandas.DataFrame
            eventnames and their pastros computed for the
            reference population

        recovered_injections: pandas.DataFrame 
            parameters of injections recovered by search
            in injection campaign

        obs_run: str
            Example: 'O3a' or 'O3b'

        ifar_threshold: float
            threshold on ifar, same as the
            threshold used on events

        """
        self.n_inj = n_inj
        # self.recovered_injections = recovered_injections
        self.pastro_ref = pastro_ref
        self.t_obs = t_obs
        self.z = Z

        mask_ifar_threshold = np.where(recovered_injections['ifar']>=ifar_threshold)[0]
        self.recovered_injections = recovered_injections.iloc[mask_ifar_threshold]

    @classmethod
    def from_hdf5(cls, file_path=None, obs_run="O3a"):
        '''
        ....
        '''
        if file_path is None:
            file_path = SUMMARY_FILE_PATHS[obs_run]
        recovered_injections_h5 = pd.DataFrame()
        pastro_ref_h5 = pd.DataFrame()
        try:
            with h5py.File(file_path, 'r') as f:
                n_inj_h5 = f['Ninj'][()]
                t_obs_h5 = f['TOBS'][()]
                pastro_ref_group = f['pastro']
                for evname, pastro in pastro_ref_group.items():
                    pastro_ref_h5[evname] = pastro[()]
                recovered_injections_group = f['recovered_injections']
                for name, dataset in recovered_injections_group.items():
                    recovered_injections_h5[name] = dataset[:]
        except KeyError as e:
            print(e)
            print(f"{file_name} does not contain all the information needed to create this object")

        return cls(n_inj=n_inj_h5, t_obs=t_obs_h5, pastro_ref=pastro_ref_h5,
                   recovered_injections=recovered_injections_h5, obs_run=obs_run)
    