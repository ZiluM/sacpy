from .linger_cal import linear_reg, multi_linreg, multi_corr, partial_corr
import numpy as np
import xarray as xr
from scipy.spatial.distance import cdist
import scipy.stats as sts
from .Util import _correct_type, _correct_type0
from copy import copy, deepcopy

EPS = 1e-6


def gen_dataarray(data, coords):
    return xr.DataArray(data=data, coords=coords)


class LinReg:
    name = "Linear Regression"
    """
    Simple linear regression
        y[idx] = slope[idx] * x + intcp[idx]
    Attributes:
        slope (np.ndarray): Slope, shape = [*number]
        intcpt (np.ndarray) : intercept , shape = [*number]
        corr (np.ndarray) : Simple correlation coefficient , shape = [*number]
        p_value (np.ndarray) : T test p value , shape = [*number]
    """

    def __init__(self, x: np.ndarray or xr.DataArray, y: np.ndarray or xr.DataArray):
        """ Simple linear regression y[idx] = slope[idx] * x + intcp[idx]

        Args:
            x (np.ndarray): shape = (time,)
            y (np.ndarray): shape = (time,*number)
            x's dim0 must equal to y'dim0 !
        """
        # if isinstance(x, xr.DataArray):
        #     x = np.array(x)
        # if isinstance(y, xr.DataArray):
        #     y = np.array(y)
        x, xdims, xcoords = _correct_type0(x)
        y, ydims, ycoords = _correct_type0(y)
        self.xdims, self.xcoords = xdims, xcoords
        self.ydims, self.ycoords = ydims, ycoords

        # judge x.dim[0] and y.dim[0] length
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x.dim0 is {x.shape[0]} , y.dim0 is {y.shape[0]} ; not equal.")
        self.x = x
        self.y = y

        slope, intcpt, corr, p_value = linear_reg(x, y)

        # judge dataarray trans
        if ydims is not None and ycoords is not None:
            ycoords_m1 = {ydims[i]: ycoords[ydims[i]] for i in range(1, len(ydims))}
            slope, intcpt, corr, p_value = map(lambda data: gen_dataarray(data, ycoords_m1),
                                               [slope, intcpt, corr, p_value])
            # slope = gen_dataarray(slope, ycoords_m1)
        self.slope, self.intcpt, self.corr, self.p_value = \
            slope, intcpt, corr, p_value

    def mask(self, threshold=0.05):
        masked = [data.copy() for data in [self.slope, self.intcpt, self.corr]]
        if self.ydims is not None and self.ycoords is not None:  # xarray term
            for i in range(3):
                masked[i] = xr.where(self.p_value <= threshold, masked[i], np.NAN)
        else:  # numpy
            for i in range(3):
                masked[i][self.p_value > threshold] = np.NAN
        self.slope1, self.intcpt1, self.corr1 = masked
        self.masked = True


    def __repr__(self) -> str:
        res = f"{self.name}, x.shape = {self.x.shape}, y.shape = {self.y.shape} "
        return res

    __str__ = __repr__


class SpaceCorr():
    name = "Space Correlation"

    def __init__(self, x, y) -> None:
        num = x.shape[0]
        self.origin_shape = x.shape
        x0 = x.reshape((num, -1)).T
        y0 = y.reshape((num, -1)).T
        # corr = 1 - cdist(x, y, "correlation")
        covar = x0.T @ y0 / (num - 1)
        corr = covar / y0.std(axis=1) / x0.std(axis=1)
        self.corr = corr.reshape(self.origin_shape)


class M2mLinReg():
    name = "Multi2Multi_Linear_Regression"

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """ calculate correlation of x[time,n_factor] and y[time,n_factor] and their p_value
        Args:
            x (np.ndarray): shape [time,n_factor]
            y (np.ndarray): shape [time,m_factor]

        Attribute:
            self.corr: [n_factor, m_factor]
            self.p_value [n_factor, m_factor]
        """

        x, xdims, xcoords = _correct_type0(x)
        y, ydims, ycoords = _correct_type0(y)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x.dim0 is {x.shape[0]} , y.dim0 is {y.shape[0]} ; not equal.")
        Num0 = x.shape[0]
        x = x.reshape((x.shape[0], -1)).T
        y = y.reshape((y.shape[0], -1)).T
        corr = 1 - cdist(x, y, "correlation")
        self.corr = corr
        t = corr / (np.sqrt(1 - corr**2) + EPS) * np.sqrt(Num0 - 2)
        p_value = sts.t.sf(t, df=Num0 - 2)
        pv_cp = np.copy(p_value)
        p_value[pv_cp >= 0.5] = (1 - p_value[pv_cp >= 0.5]) * 2
        p_value[pv_cp < 0.5] = (p_value[pv_cp < 0.5]) * 2
        self.p_value = p_value



class MultLinReg:
    name = "MultLinReg"
    """
    multiple linear regression ::
        y[idx] = slope[idx,0] * x[0] + slope[idx, 1] * x[1] + ... + intcp[idx]
    Attribute:
        slope (np.ndarray): Slope, shape = [N,*number]
        intcpt (np.ndarray) : intercept , shape = [*number]
        R (np.ndarray) :
            multiple correlation coefficient , shape = [*number]
        pv_all(np.ndarray) : F test p value , shape = [*number]
        pv_i(np.ndarray) :
            F test p value of every infact, shape = [N, *number]
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, cal_sim=True):
        """
        Args:
            x (np.ndarray): shape = (time, N) ; N is Number of factors
            y (np.ndarray): shape = (time,*number)
            cal_sim (Bool) : Whether to call function multi_linreg
            x's dim0 must equal to y'dim0 !
        """
        # if isinstance(x, xr.DataArray):
        #     x = np.array(x)
        # if isinstance(y, xr.DataArray):
        #     y = np.array(y)
        x, xdims, xcoords = _correct_type0(x)
        y, ydims, ycoords = _correct_type0(y)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x.dim0 is {x.shape[0]} , y.dim0 is {y.shape[0]} ; not equal.")
        self.x = x
        self.y = y
        if cal_sim:
            self.slope, self.intcpt, self.R, self.pv_all, self.pv_i = multi_linreg(x, y)
        else:
            self.slope, self.intcpt, self.R, self.pv_all, self.pv_i = None, None, None, None

        self.multi_corr = None
        self.part_corr = {}

    def cal_corr(self):
        """ cal correlation between X_factors , y
        Returns:
            correlation : shape = [*y.shape[1:],Numf+1,Numf+1]
        """
        self.multi_corr = multi_corr(self.x, self.y)
        return self.multi_corr

    def cal_paritial_corr(self, idx: int):
        """calculate partial correlation 
        Args:
            idx (int): correlation between y and x[:,idx], Exclude the influence of other factors
        """
        self.part_corr[idx] = partial_corr(self.x, self.y, idx, mul_corr=self.multi_corr)
        return self.part_corr[idx]

    # def _repr_html_(self):
    #     """
    #     show in jupyter
    #     """
    #     from .repr_html import res_repr_html
    #     return (res_repr_html(self))

    def __repr__(self) -> str:
        res = f"{self.name}, x.shape = {self.x.shape}, y.shape = {self.y.shape} "
        return res

    __str__ = __repr__
