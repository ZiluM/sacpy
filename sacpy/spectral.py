import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
import scipy.stats as stats


def autocorr(x):
    """
    Find the autocorrelation of a timeseries `x`.

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    acor : float
    """
    xd = (x - np.mean(x)) / np.std(x)
    acor = np.dot(xd[0:len(xd)-1], xd[1:len(x)]) / (len(xd)-1)  # I am not sure this should be N or N-1; 
    # Dennis' code seems to use N-1, but his textbook says N
    return acor


def harmonic_func(n, period=365.25, num_fs=4):
    """
    Construct a harmonic function for regression.

    Parameters
    ----------
    n : int
        The sampling dimension obtained from the original data.

    period : float
        The period of the regression function.

    num_fs : int
        The number of frequency bands to use.

    Returns
    -------
    func : np.ndarray
        The matrix to regress on to the original timeseries.
    """
    func = np.zeros((num_fs*2+1, n), dtype=float)
    time = np.arange(0, n) * 2 * np.pi / period
    func[0, :] = np.ones(n)
    for i in range(num_fs):
        func[2*i+1, :] = np.sin(i * time)
        func[(i+1)*2, :] = np.cos(i * time)
    return func


def prewhiten(x, ac):
    """Prewhiten a time series `x` with a predetermined autocorrelation factor `ac`."""
    # Do not wish to modify x in place, so make a copy
    x_copy = x.copy()
    x_copy[1:] -= ac * x_copy[:len(x)-1]
    return x_copy


def red_shape(freq, ac, factor):
    """
    Calculate the curve for red shape.

    Parameters
    ----------
    TODO
    """
    rs = factor * (1.0 - ac ** 2) / (1. - (2.0 * ac * np.cos(freq * 2.0 * np.pi)) + ac ** 2)
    return rs


def get_fcrit(p_crit, dfn, dfd):
    """
    Simply iterate through to get a critical value for f. Taken from Dennis' code.
    """
    for n in range(200):
        f = 1. + float(n) / 50.
        p_val = stats.f.cdf(f, dfn, dfd)
        if p_val > p_crit:
            f_crit = f
            break
    return f_crit


def cohstat(dof, siglev):
    """Coherence significance level. Not ideal as the numbers are hard coded; there must be a more elegant way of doing this."""
    f99 = [0.99,0.684,0.602,0.536,0.482,0.438,0.401,0.342,0.264,0.215,0.175,0.147,0.112,0.075,0.057,0.045,0.023,0.002]
    f90 = [0.901,0.437,0.370,0.319,0.280,0.250,0.226,0.189,0.142,0.112,0.091,0.076,0.057,0.038,0.029,0.023,0.011,0.001]
    f95 = [0.951,0.527,0.450,0.393,0.348,0.312,0.283,0.238,0.181,0.146,0.118,0.098,0.074,0.050,0.037,0.030,0.015,0.001]
    n = [2,5,6,7,8,9,10,12,16,20,25,30,40,60,80,100,200,1000000]

    if siglev > 0.95:
        f = f99
    else:
        f = f95
    coh_crit = np.interp(dof, n, f)
    return coh_crit


class Spectral():
    """TODO
    
    Notes
    -----
    `CrossSpectral` should be a subclass of `Spectral`.
    """
    def __init__(self, x, M_length, 
                 overlap=0.5,
                 remove_trend=True,
                 remove_annual=True, 
                 normalize_series=True, 
                 prewhiten=False) -> None:
        """
        Parameters
        ----------
        TODO

        Notes
        -----
        If only a segment from each year is used, `overlap` should be set to 0.
        """
        self.x = x

        self.dfn = 2.0 * len(self.x) / M_length   # DOF of spectrum conservative estimate (abbreviated as the degree of freedom for the numerator, dfn)
        self.dfd = len(self.x) / 2  # DOF of denominator Null Hypothesis (abbreviated as the degree of freedom for the denominator, dfd)

        self.M_length = M_length
        self.overlap = overlap

        self._preprocess_pipeline(remove_trend, remove_annual, normalize_series, prewhiten)

        self.ac_x = autocorr(self.x)  # compute the autocorrelation after the preprocess pipeline

    def _detrend(self):
        """Detrend the time series."""
        self.x = signal.detrend(self.x)

    def _normalize(self):
        """Normalize the time series to make their variance equal to 1."""
        self.x /= np.std(self.x)
    
    def _remove_annual(self):
        """Remove the annual trend."""
        Nx = len(self.x)
        func = harmonic_func(len(self.x))
        for _ in range(3):
            cx = func @ self.x / Nx
            self.x -= cx.data @ func
    
    def _prewhiten(self):
        """TODO"""
        self.x = prewhiten(self.x)

    def _preprocess_pipeline(self, remove_trend, remove_annual, normalize_series, prewhiten) -> None:
        """TODO"""
        if remove_trend:
            self._detrend()
        if remove_annual:
            self._remove_annual()
        if normalize_series:
            self._normalize()
        if prewhiten:
            self._prewhiten()

    def get_spectra(self, normalize_spectrum=True, detrend='linear', **kwargs):
        """
        Compute spectrum for `x`.

        Paramaters
        ----------
        TODO

        Returns
        -------
        TODO

        Notes
        -----
        **kwargs are directly passed into the functional call of `signal.welch`.
        """
        f, Pxx = signal.welch(self.x, nperseg=self.M_length, noverlap=self.M_length * self.overlap, detrend=detrend, **kwargs)

        if normalize_spectrum:
            Pxx /= np.mean(Pxx)

        self.f = f
        self.Pxx = Pxx

        return self.f, self.Pxx
    
    def get_red_noise(self, method='fit_theory'):
        """
        Fit the x and y spectra with red noise.

        Parameters
        ----------
        method : str
            Should be one of 'fit_theory' or 'fit_data'.

        Returns
        -------
        TODO
        """
        # Check if freq exists, if not compute it
        if not hasattr(self, 'f'):
            _ = self.get_spectra()

        if method == 'fit_theory':
            rsx = red_shape(self.f, self.ac_x, 1.0)
            self.rsx = rsx * np.sum(self.Pxx[1:]) / np.sum(rsx[1:])
        elif method == 'fit_data':
            param_x, _ = curve_fit(red_shape, self.f, self.Pxx, p0=(0.5, 1.))
            a1, a2 = param_x[0], param_x[1]
            
            self.rsx = red_shape(self.f, a1, a2)
        else:
            raise ValueError('"method" should be one of "fit_theory" or "fit_data."')
        
        return self.rsx
    
    def get_rs_sig(self, p_crit=0.99):
        """Get the statistical significance curve based on red noise theory.
        
        Returns
        -------
        rsx_sig, rsy_sig : np.ndarray, np.ndarray
        """
        self.f_crit = get_fcrit(p_crit, self.dfn, self.dfd)

        if not hasattr(self, 'rsx'):
            _ = self.get_red_noise()

        self.rsx_sig = self.f_crit * self.rsx

        return self.rsx_sig


class CrossSpectral(Spectral):
    """
    Perform a cross spectral analysis on two timeseries `x` and `y`.
    """
    def __init__(self, x, y, M_length, 
                 overlap=0.5,
                 remove_trend=True, 
                 remove_annual=True,
                 normalize_series=True,
                 prewhiten=False) -> None:
        """
        Parameters
        ----------
        x, y : array-like

        TODO
        """
        self.y = y
        super().__init__(x, M_length, overlap, remove_trend, remove_annual, normalize_series, prewhiten)
        self.ac_y = autocorr(self.y)

    def _detrend(self):
        """Detrend the time series."""
        super()._detrend()
        self.y = signal.detrend(self.y)

    def _normalize(self):
        """Normalize the time series to make their variance equal to 1."""
        super()._normalize()
        self.y /= np.std(self.y)
    
    def _remove_annual(self):
        """Remove the annual trend."""
        Nx, Ny = len(self.x), len(self.y)
        func = harmonic_func(len(self.x))
        for _ in range(3):
            cx = func @ self.x / Nx
            cy = func @ self.y / Ny
            self.x -= cx.data @ func
            self.y -= cy.data @ func
    
    def _prewhiten(self):
        """TODO"""
        super()._prewhiten()
        self.y = prewhiten(self.y)
    
    def get_spectra(self, normalize_spectrum=True, detrend='linear', **kwargs):
        """
        Compute spectra for `x` and `y`.

        Paramaters
        ----------
        TODO

        Returns
        -------
        TODO

        Notes
        -----
        **kwargs are directly passed into the functional call of `signal.welch`.
        """
        # May need to pop nperseg, noverlap, detrend here for robustness?
        # Assumption: f from x spectrum and y spectrum are the same
        super().get_spectra(normalize_spectrum, detrend, **kwargs)
        _, Pyy = signal.welch(self.y, nperseg=self.M_length, noverlap=self.M_length * self.overlap, detrend=detrend, **kwargs)

        if normalize_spectrum:
            Pyy /= np.mean(Pyy)

        self.Pyy = Pyy

        return self.f, self.Pxx, self.Pyy
    
    def get_red_noise(self, method='fit_theory'):
        """
        Fit the x and y spectra with red noise.

        Parameters
        ----------
        method : str
            Should be one of 'fit_theory' or 'fit_data'.

        Returns
        -------
        TODO
        """
        # Check if freq exists, if not compute it
        super().get_red_noise(method=method)

        if method == 'fit_theory':
            rsy = red_shape(self.f, self.ac_y, 1.0)
            self.rsy = rsy * np.sum(self.Pyy[1:]) / np.sum(rsy[1:])
        elif method == 'fit_data':
            param_y, _ = curve_fit(red_shape, self.f, self.Pyy, p0=(0.5, 1.))
            b1, b2 = param_y[0], param_y[1]
            self.rsy = red_shape(self.f, b1, b2)
        
        return self.rsx, self.rsy
    
    def get_rs_sig(self, p_crit=0.99):
        """Get the statistical significance curve based on red noise theory.
        
        Returns
        -------
        rsx_sig, rsy_sig : np.ndarray, np.ndarray
        """
        super().get_rs_sig(p_crit=p_crit)
        self.rsy_sig = self.rsy * self.f_crit

        return self.rsx_sig, self.rsy_sig

    def get_cross_spectra(self, detrend='linear', **kwargs):
        """
        Get the cross spectra between x and y.

        Returns
        -------
        covariance, phase, coherence : TODO
        """
        _, Pxy = signal.csd(self.x ,self.y, nperseg=self.M_length,detrend=detrend)
        _, Coh = signal.coherence(self.x, self.y, nperseg=self.M_length,detrend=detrend)

        self.covariance = np.real(Pxy)
        self.phase = np.arctan2(np.imag(Pxy), np.real(Pxy)) * 180 / np.pi
        self.coherence = Coh

        return self.covariance, self.phase, self.coherence
    
    def get_coh_sig(self, p_crit=0.99):
        """Obtain the coherence significance.
        
        Returns
        -------
        TODO
        """
        self.coh_sig = cohstat(self.dfn, p_crit)
        return self.coh_sig
