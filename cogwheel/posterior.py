"""
Define the Posterior class.
Can run as a script to make and save a Posterior instance from scratch.
"""

import argparse
import inspect
import pathlib
import sys
import os
import tempfile
import textwrap
import time
import numpy as np

from . import data
from . import gw_prior
from . import utils
from . import waveform
from . likelihood import RelativeBinningLikelihood, ReferenceWaveformFinder


class PosteriorError(Exception):
    """Error raised by the Posterior class."""


class Posterior(utils.JSONMixin):
    """
    Class that instantiates a prior and a likelihood and provides
    methods for sampling the posterior distribution.

    Parameter space folding is implemented; this means that some
    ("folded") dimensions are sampled over half their original range,
    and a map to the other half of the range is defined by reflecting
    about the midpoint. The folded posterior distribution is defined as
    the sum of the original posterior over all `2**n_folds` mapped
    points. This is intended to reduce the number of modes in the
    posterior.
    """
    def __init__(self, prior, likelihood):
        """
        Parameters
        ----------
        prior:
            Instance of `prior.Prior`, provides coordinate
            transformations, priors, and foldable parameters.
        likelihood:
            Instance of `likelihood.RelativeBinningLikelihood`, provides
            likelihood computation.
        """
        if set(prior.standard_params) != set(
                likelihood.waveform_generator.params):
            raise PosteriorError('The prior and likelihood instances passed '
                                 'have incompatible parameters.')

        self.prior = prior
        self.likelihood = likelihood

        # Increase `n_cached_waveforms` to ensure fast moves remain fast
        fast_sampled_params = self.prior.get_fast_sampled_params(
            self.likelihood.waveform_generator.fast_params)
        n_slow_folded = len(set(self.prior.folded_params)
                            - set(fast_sampled_params))
        self.likelihood.waveform_generator.n_cached_waveforms \
            = 2 ** n_slow_folded

        # Match lnposterior signature to that of transform
        self.lnposterior.__func__.__signature__ = inspect.signature(
            self.prior.__class__.transform)

    def lnposterior(self, *args, **kwargs):
        """
        Natural logarithm of the posterior probability density in
        the space of sampled parameters (does not apply folding).
        """
        lnprior, standard_par_dic = self.prior.lnprior_and_transform(
            *args, **kwargs)
        return lnprior + self.likelihood.lnlike(standard_par_dic)

    @classmethod
    def from_event(cls, event, approximant, prior_class, fbin=None,
                   pn_phase_tol=.05, disable_precession=False,
                   harmonic_modes=None, tolerance_params=None, seed=0,
                   tc_rng=(-.1, .1), **kwargs):
        """
        Instantiate a `Posterior` class from the strain data.
        Automatically find a good fit solution for relative binning.

        Parameters
        ----------
        event: Instance of `data.EventData` or string with
               event name.
        approximant: string with approximant name.
        prior_class: string with key from `gw_prior.prior_registry`,
                     or subclass of `prior.Prior`.
        fbin: Array with edges of the frequency bins used for relative
              binning [Hz]. Alternatively, pass `pn_phase_tol`.
        pn_phase_tol: Tolerance in the post-Newtonian phase [rad] used
                      for defining frequency bins. Alternatively, pass
                      `fbin`.
        disable_precession: bool, whether to set inplane spins to 0
                            when evaluating the waveform.
        harmonic_modes: list of 2-tuples with (l, m) of the harmonic
                        modes to include. Pass `None` to use
                        approximant-dependent defaults per
                        `waveform.APPROXIMANTS`.
        tolerance_params: dictionary
        kwargs: Additional keyword arguments to instantiate the prior
                class.

        Return
        ------
        Instance of `Posterior`.
        """
        if isinstance(event, data.EventData):
            event_data = event
        elif isinstance(event, str):
            event_data = data.EventData.from_npz(event)
        else:
            raise ValueError('`event` must be of type `str` or `EventData`')

        if isinstance(prior_class, str):
            prior_class = gw_prior.prior_registry[prior_class]

        # Check required input before doing expensive maximization:
        required_pars = {
            parameter.name for parameter in prior_class.init_parameters()[1:]
            if parameter.default is inspect._empty
            and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                       inspect.Parameter.VAR_KEYWORD)}
        event_data_keys = {'mchirp_range', 'tgps', 'q_min'}
        bestfit_keys = {'ref_det_name', 'detector_pair', 'f_ref', 't0_refdet'}
        if missing_pars := (required_pars - event_data_keys - bestfit_keys
                            - set(kwargs)):
            raise ValueError(f'Missing parameters: {", ".join(missing_pars)}')

        # Initialize likelihood:
        aux_waveform_generator = waveform.WaveformGenerator(
            event_data.detector_names, event_data.tgps, event_data.tcoarse,
            approximant, f_ref=20., harmonic_modes=[(2, 2)])
        bestfit = ReferenceWaveformFinder(
            event_data, aux_waveform_generator).find_bestfit_pars(tc_rng, seed)
        waveform_generator = waveform.WaveformGenerator(
            event_data.detector_names, event_data.tgps, event_data.tcoarse,
            approximant, bestfit['f_ref'], harmonic_modes, disable_precession)
        likelihood = RelativeBinningLikelihood(
            event_data, waveform_generator, bestfit['par_dic'], fbin,
            pn_phase_tol, tolerance_params)
        assert likelihood._lnl_0 > 0

        # Initialize prior:
        prior = prior_class(**
            {key: getattr(event_data, key) for key in event_data_keys}
            | bestfit | kwargs)

        # Initialize posterior and do second search:
        posterior = cls(prior, likelihood)
        posterior.refine_reference_waveform(seed)
        return posterior

    def refine_reference_waveform(self, seed=None):
        """
        Reset relative-binning reference waveform, using differential
        evolution to find a good fit.
        It is guaranteed that the new waveform will have at least as
        good a fit as the current one.
        The likelihood maximization uses folded sampled parameters.
        """
        folded_par_vals_0 = self.prior.fold(
            **self.prior.inverse_transform(**self.likelihood.par_dic_0))

        lnlike_unfolds = self.prior.unfold_apply(
            lambda *pars: self.likelihood.lnlike(self.prior.transform(*pars)))

        bestfit_folded = utils.differential_evolution_with_guesses(
            func=lambda pars: -max(lnlike_unfolds(*pars)),
            bounds=list(zip(self.prior.cubemin,
                            self.prior.cubemin + self.prior.folded_cubesize)),
            guesses=folded_par_vals_0,
            seed=seed).x
        i_fold = np.argmax(lnlike_unfolds(*bestfit_folded))

        self.likelihood.par_dic_0 = self.prior.transform(
            *self.prior.unfold(bestfit_folded)[i_fold])
        print(f'Found solution with lnl = {self.likelihood._lnl_0}')

    def get_eventdir(self, parentdir):
        """
        Return directory name in which the Posterior instance
        should be saved, of the form
        {parentdir}/{prior_name}/{eventname}/
        """
        return utils.get_eventdir(parentdir, self.prior.__class__.__name__,
                                  self.likelihood.event_data.eventname)


def initialize_posteriors_slurm(eventnames, approximant, prior_name,
                                parentdir, n_hours_limit=2,
                                memory_per_task='4G', overwrite=False):
    """
    Submit jobs that initialize `Posterior.from_event()` for each event.
    """
    package = pathlib.Path(__file__).parents[1].resolve()
    module = f'cogwheel.{os.path.basename(__file__)}'.rstrip('.py')

    if isinstance(eventnames, str):
        eventnames = [eventnames]
    for eventname in eventnames:
        eventdir = utils.get_eventdir(parentdir, prior_name, eventname)

        if not overwrite and (filename := eventdir/'Posterior.json').exists():
            raise FileExistsError(
                f'{filename} exists, pass `overwrite=True` to overwrite.')

        utils.mkdirs(eventdir)

        job_name = f'{eventname}_posterior'
        stdout_path = (eventdir/'posterior_from_event.out').resolve()
        stderr_path = (eventdir/'posterior_from_event.err').resolve()

        args = ' '.join([eventname, approximant, prior_name, parentdir])
        if overwrite:
            args += ' --overwrite'

        with tempfile.NamedTemporaryFile('w+') as batchfile:
            batchfile.write(textwrap.dedent(f"""\
                #!/bin/bash
                #SBATCH --job-name={job_name}
                #SBATCH --output={stdout_path}
                #SBATCH --error={stderr_path}
                #SBATCH --mem-per-cpu={memory_per_task}
                #SBATCH --time={n_hours_limit:02}:00:00

                eval "$(conda shell.bash hook)"
                conda activate {os.environ['CONDA_DEFAULT_ENV']}

                cd {package}
                srun {sys.executable} -m {module} {args}
                """))
            batchfile.seek(0)

            os.system(f'chmod 777 {batchfile.name}')
            os.system(f'sbatch {batchfile.name}')
            time.sleep(.1)


def main(eventname, approximant, prior_name, parentdir, overwrite):
    '''Construct a Posterior instance and save it to json.'''
    post = Posterior.from_event(eventname, approximant, prior_name)
    post.to_json(post.get_eventdir(parentdir), overwrite=overwrite)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Construct a Posterior instance and save it to json.''')

    parser.add_argument('eventname', help='key from `data.event_registry`.')
    parser.add_argument('approximant', help='key from `waveform.APPROXIMANTS`')
    parser.add_argument('prior_name',
                        help='key from `gw_prior.prior_registry`')
    parser.add_argument('parentdir', help='top directory to save output')
    parser.add_argument('--overwrite', action='store_true',
                        help='pass to overwrite existing json file')

    main(**vars(parser.parse_args()))
