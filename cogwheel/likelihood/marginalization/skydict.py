"""
Implement class ``SkyDictionary``, useful for marginalizing over sky
location.
"""

import collections
import numpy as np
import scipy.signal
from scipy.stats import qmc

from cogwheel import gw_utils
from cogwheel import utils


class SkyDictionary(utils.JSONMixin):
    """
    Given a network of detectors, this class generates a set of
    samples covering the sky location isotropically in Earth-fixed
    coordinates (lat, lon).
    The samples are assigned to bins based on the arrival-time delays
    between detectors. This information is accessible as dictionaries
    ``delays2inds_map``, ``delays2genind_map``.
    Antenna coefficients F+, Fx (psi=0) and detector time delays from
    geocenter are computed and stored for all samples.
    """
    def __init__(self, detector_names, *, f_sampling: int = 2**13,
                 nsky: int = 10**6, seed=0):
        self.detector_names = tuple(detector_names)
        self.nsky = nsky
        self.f_sampling = f_sampling
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.sky_samples = self._create_sky_samples()
        self.fplus_fcross_0 = gw_utils.get_fplus_fcross_0(self.detector_names,
                                                          **self.sky_samples)
        geocenter_delays = gw_utils.get_geocenter_delays(
            self.detector_names, **self.sky_samples)
        self.geocenter_delay_first_det = geocenter_delays[0]
        self.delays = geocenter_delays[1:] - geocenter_delays[0]

        self.delays2inds_map = self._create_delays2inds_map()
        self.delays2genind_map = {
            delays_key: self._create_index_generator(inds)
            for delays_key, inds in self.delays2inds_map.items()}

    def resample_timeseries(self, timeseries, times, axis=-1):
        """
        Resample a timeseries to match the SkyDict's sampling frequency.
        The sampling frequencies of the SkyDict and ``timeseries`` must
        be multiples (or ``ValueError`` is raised).

        Parameters
        ----------
        timeseries: array_like
            The data to resample.

        times: array_like
             Equally-spaced sample positions associated with the signal
             data in `timeseries`.

        axis: int
            The axis of timeseries that is resampled. Default is -1.

        Return
        ------
        resampled_timeseries, resampled_times
            A tuple containing the resampled array and the corresponding
            resampled positions.
        """
        fs_ratio = self.f_sampling * (times[1] - times[0])
        if fs_ratio != 1:
            timeseries, times = scipy.signal.resample(
                timeseries, int(len(times) * fs_ratio), times, axis=axis)
            if not np.isclose(1 / self.f_sampling, times[1] - times[0]):
                raise ValueError(
                    '`times` is incommensurate with `f_sampling`.')

        return timeseries, times

    def get_sky_inds_and_prior(self, delays):
        """
        Parameters
        ----------
        delays: int array of shape (n_det-1, n_samples)
            Time-of-arrival delays in units of 1 / self.f_sampling

        Return
        ------
        sky_inds: tuple of ints of length n_samples
            Indices of self.sky_samples with the correct time delays.

        sky_prior: tuple of floats of length n_samples
            Prior probability density for the time-delays, in units of
            s^-(n_det-1).
        """
        return zip(*(next(self.delays2genind_map[delays_key])
                     for delays_key in zip(*delays)))

    def _create_sky_samples(self):
        """
        Return a dictionary of samples in terms of 'lat' and 'lon' drawn
        isotropically by means of a Quasi Monte Carlo (Halton) sequence.
        """
        samples = {}
        u_lat, u_lon = qmc.Halton(2, seed=self._rng).random(self.nsky).T

        samples['lat'] = np.arcsin(2*u_lat - 1)
        samples['lon'] = 2 * np.pi * u_lon
        return samples

    def _create_delays2inds_map(self):
        """
        Return a dictionary mapping arrival time delays to sky-sample
        indices.
        Its keys are tuples of ints of length (n_det - 1), with time
        delays to the first detector in units of 1/self.f_sampling.
        Its values are list of indices to ``self.sky_samples`` of
        samples that have the corresponding (discretized) time delays.
        """
        # (ndet-1, nsky)
        delays_keys = zip(*np.rint(self.delays * self.f_sampling).astype(int))

        delays2inds_map = collections.defaultdict(list)
        for i_sample, delays_key in enumerate(delays_keys):
            delays2inds_map[delays_key].append(i_sample)

        return delays2inds_map

    def _create_index_generator(self, inds):
        """
        Infinite generator that yields pairs of
        (i_sample, delays_key_prior).

        Parameters
        ----------
        inds: list of int
            Indices of samples in a single time-delays bin.

        Yield
        -----
        i_sample: int
            Loops infinitely over ``inds``.

        delays_key_prior: float
            Always the same constant, prior probability density for the
            particular time-delays bins, in units of s^-(n_det-1).
        """
        delays_key_prior = (self.f_sampling**(len(self.detector_names) - 1)
                            * len(inds) / self.nsky)
        while True:
            for i_sample in inds:
                yield i_sample, delays_key_prior
