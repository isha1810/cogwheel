"""
Get samples, likelihood, and prior objects from parameter estimation run
after post-processing for data analysis and visualization.
"""
import numpy as np
import os
import pathlib
import sys
import json
import pandas as pd
from copy import deepcopy as dcopy

from . import parameter_aliasing as aliasing
from . import parameter_label_formatting as label_formatting
from . import standard_intrinsic_transformations as pxform
from . import pe_plotting as peplot

COGWHEEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'cogwheel'))
sys.path.append(COGWHEEL_PATH)
from cogwheel import utils
from cogwheel import sampling
from cogwheel import grid as gd
from cogwheel import cosmology as cosmo
from cogwheel import postprocessing


class AnalysisHandle:
    """Class for analyzing posteriors."""
    LNL_COL = 'lnl'
    KEYMAP = aliasing.PARKEY_MAP
    PAR_LABELS = label_formatting.param_labels
    PAR_UNITS = label_formatting.units
    PAR_NAMES = label_formatting.param_names
    
    def __init__(self, rundir, name=None):
        super().__init__()
        # associate to run directory
        self.rundir = pathlib.Path(rundir)
        self.name = name or self.rundir.parts[-1]

        # load posterior attributes
        sampler = utils.read_json(self.rundir / sampling.Sampler.JSON_FILENAME)
        self.likelihood = dcopy(sampler.posterior.likelihood)
        self.prior = dcopy(sampler.posterior.prior)
        # make these references for direct access
        self.evdata = self.likelihood.event_data
        self.wfgen = self.likelihood.waveform_generator
        self.evname = self.evdata.eventname

        # load samples
        self.samples_path = self.rundir/sampling.SAMPLES_FILENAME
        self.samples = pd.read_feather(self.samples_path)

        # check likelihood information
        if self.LNL_COL not in self.samples:
            self.LNL_COL = self.key(self.LNL_COL)
        if self.LNL_COL not in self.samples:
            print('WARNING: No likelihood information found in samples.')
            self.best_par_dic = None
        else:
            self.best_par_dic = self.get_best_par_dics()

        # see if the keymap is faithful to samples
        if not all([self.key(k) == k for k in self.samples.columns]):
            print('WARNING: Abandoning self.KEYMAP due to inconsistency with samples.')
            self.KEYMAP = {}

        #load test results if they're there
        self.tests_path = self.rundir/postprocessing.TESTS_FILENAME
        self.tests_dict = None
        if os.path.isfile(self.tests_path):
            self.tests_dict = json.load(open(self.tests_path, 'r'))


    def key(self, key):
        return self.KEYMAP.get(key, key)

    def par_label(self, key):
        return self.PAR_LABELS.get(self.key(key), key)

    def par_name(self, key):
        return self.PAR_NAMES.get(self.key(key), key)

    def par_unit(self, key):
        return self.PAR_UNITS.get(self.key(key), key)

    def get_best_par_dics(self, key_rngs={}, get_best_inds=0, as_par_dics=True):
        s = self.samples[np.isnan(self.samples[self.LNL_COL]) == False]
        for k, rng in key_rngs.items():
            s = s[s[self.key(k)] > rng[0]]
            s = s[s[self.key(k)] < rng[1]]
        s = s.sort_values(self.LNL_COL, ascending=False).reset_index().iloc[get_best_inds]
        if as_par_dics:
            s = s[[self.key(k) for k in self.wfgen.params]]
            if hasattr(get_best_inds, '__len__'):
                return [dict(idx_row[1]) for idx_row in s.iterrows()]
            return dict(s)
        return s

    def add_source_parameters(self, redshift_key=None, mass_keys=['m1', 'm2', 'mtot', 'mchirp']):
        """
        Add _source version of each mass in mass_keys using *= 1+self.samples[redshift_key].
        If redshift_key is None, do intrinsic parameter completion with pxform.compute_samples_aux_vars
        """
        if redshift_key is None:
            # this completes intrinsic parameter space and adds redshift and source frame information
            pxform.compute_samples_aux_vars(self.samples)
            return
        rkey = self.key(redshift_key)
        if rkey not in self.samples:
            self.samples[rkey] = cosmo.z_of_DL_Mpc(self.samples['d_luminosity'])
        for k in mass_keys:
            self.samples[self.key(k)+'_source'] = self.samples[self.key(k)] / (1+self.samples[rkey])

    def write_complete_samples(self, fname=None, overwrite=False, antenna=False,
                               cosmo_weights=False, ligo_angles=False):
        if fname is None:
            fname = self.samples_path
        if os.path.exists(fname) and (not overwrite):
            raise FileExistsError(f'Set overwrite=True to overwrite {fname}')
        self.add_source_parameters()
        if ligo_angles:
            self.add_ligo_angles()
        if antenna:
            self.add_antenna()
        if cosmo_weights:
            self.add_cosmo_weights()
        self.samples.to_feather(self.samples_path)

    def add_ligo_angles(self, keep_new_spins=False):
        self.samples = peplot.samples_with_ligo_angles(self.samples, self.wfgen.f_ref,
                                                       keep_new_spins=keep_new_spins)

    def add_antenna(self):
        peplot.samples_add_antenna_response(self.samples, det_chars=self.evdata.detector_names,
                                            tgps=self.evdata.tgps)

    def add_cosmo_weights(self):
        peplot.samples_add_cosmo_prior(self.samples)


    def corner_plot(self, parkeys=['mchirp', 'q', 'chieff'], weights=None,
                    extra_grid_kwargs={}, **corner_plot_kwargs):
        """
        Make corner plot of self.samples for the parameter keys in parkeys.

        **corner_plot_kwargs can include anything (except pdf) from
          Grid.corner_plot(pdf=None, title=None, subplot_size=2., fig=None, ax=None,
                figsize=None, nbins=6, set_legend=True, save_as=None, y_title=.98,
                plotstyle=None, show_titles_1d=True, scatter_points=None, **kwargs)
        --> NAMELY, pass fig=myfig, ax=myax to plot with existing axes

        weights can be an array of weights or a key to use from self.samples
        scatter_points (corner_plot_kwargs) can be DataFrame of extra samples to plot
        """
        if isinstance(weights, str):
            weights = self.samples[self.key(weights)]
        pdfnm = f'{self.evname}: {self.name}\n{len(self.samples)} samples'
        corner_plot_kwargs['set_legend'] = corner_plot_kwargs.get('set_legend', True)
        if 'title' not in corner_plot_kwargs:
            corner_plot_kwargs['title'] = pdfnm
            pdfnm = None
        return gd.Grid.from_samples([self.key(k) for k in parkeys],
            self.samples, pdf_key=pdfnm, units=self.PAR_UNITS,
            labels=self.PAR_LABELS, weights=weights,
            **extra_grid_kwargs).corner_plot(pdf=pdfnm, **corner_plot_kwargs)

    def plot_psd(self, ax=None, fig=None, label=None, plot_type='loglog',
                 weights=None, plot_asd=False, xlim=None, ylim=None, title=None,
                 figsize=None, use_fmask=False, **plot_kws):
        msk = (self.evdata.fslice if use_fmask else slice(None))
        dets_xplot = self.evdata.frequencies[msk]
        dets_yplot = self.evdata.psd[..., msk]
        ylabel = 'Power Spectral Density'
        if plot_asd:
            dets_yplot = np.sqrt(dets_yplot)
            ylabel = 'Amplitude Spectral Density'
        if weights is not None:
            dets_yplot *= weights[msk]
            ylabel = 'Weighted ' + ylabel
        return peplot.plot_at_dets(dets_xplot, dets_yplot, ax=ax, fig=fig, label=label,
                                   xlabel='Frequency (Hz)', ylabel=ylabel, plot_type=plot_type,
                                   xlim=xlim, ylim=ylim, title=title, det_names=self.evdata.detector_names,
                                   figsize=figsize, **plot_kws)

    def plot_wf_amp(self, par_dic, whiten=True, by_m=False, ax=None, fig=None, label=None,
                    plot_type='loglog', weights=None, xlim=None, ylim=None,
                    title=None, figsize=None, use_fmask=False, **plot_kws):
        msk = (self.evdata.fslice if use_fmask else slice(None))
        dets_xplot = self.evdata.frequencies[msk]
        h_f = self.likelihood._get_h_f(par_dic, by_m=by_m)
        if whiten:
            h_f = self.evdata.dt * np.fft.rfft(self.likelihood._get_whitened_td(h_f), axis=-1)
        if weights is not None:
            h_f *= weights
        if by_m:
            for j, lmlist in enumerate(self.wfgen._harmonic_modes_by_m.values()):
                dets_yplot = np.abs(h_f[j, :, msk])
                fig, ax = peplot.plot_at_dets(dets_xplot, dets_yplot, ax=ax, fig=fig, label=str(lmlist),
                    xlabel='Frequency (Hz)', ylabel='Waveform Amplitude', plot_type=plot_type,
                    xlim=xlim, ylim=ylim, title=title, det_names=self.evdata.detector_names,
                    figsize=figsize, **plot_kws)
            return fig, ax
        return peplot.plot_at_dets(dets_xplot, np.abs(h_f[:, msk]), ax=ax, fig=fig, label=label,
                                   xlabel='Frequency (Hz)', ylabel='Waveform Amplitude',
                                   plot_type=plot_type, xlim=xlim, ylim=ylim, title=title,
                                   det_names=self.evdata.detector_names, figsize=figsize, **plot_kws)

    def plot_whitened_wf(self, par_dic, trng=(-.7, .1), **kwargs):
        return self.likelihood.plot_whitened_wf(par_dic, trng=trng, **kwargs)

    def plot_3d_location(self, fig=None, ax=None, ckey='lnl', nstep=2,
                         clab=None, mask_keys_min={}, mask_keys_max={},
                         extra_point_dicts=[], title=None, units='Mpc',
                         figsize=(12, 12), xlim='auto', ylim='auto', zlim='auto',
                         titlesize=20, plot_kws=None, colorbar_kws=None):
        return peplot.plot_loc3d(self.samples, title=title, xlim=xlim, ylim=ylim, zlim=zlim,
                                 nstep=nstep, ckey=ckey, clab=clab, mask_keys_min=mask_keys_min,
                                 mask_keys_max=mask_keys_max, plot_kws=plot_kws, figsize=figsize,
                                 titlesize=titlesize, colorbar_kws=colorbar_kws, units=units,
                                 extra_point_dicts=extra_point_dicts, fig=fig, ax=ax)

    def plot_inplane_spin(self, color_key='q', use_V3=False, secondary_spin=False,
                          fractions=[.5, .95], plotstyle_color='r', scatter_alpha=.5,
                          figsize=None, title=None, tight=False, **colorbar_kws):
        return peplot.plot_inplane_spin(self.samples, color_key=color_key, use_V3=use_V3,
                                        secondary_spin=secondary_spin, fractions=fractions,
                                        plotstyle_color=plotstyle_color, scatter_alpha=scatter_alpha,
                                        figsize=figsize, title=title, tight=tight, **colorbar_kws)