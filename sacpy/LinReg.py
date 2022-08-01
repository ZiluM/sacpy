from .linger_cal import linear_reg, multi_linreg, multi_corr, partial_corr
import numpy as np
import xarray as xr


class LinReg:
    """
    Simple linear regression
        y[idx] = slope[idx] * x + intcp[idx]
    Attributes:
        slope (np.ndarray): Slope, shape = [*number]
        intcpt (np.ndarray) : intercept , shape = [*number]
        corr (np.ndarray) : Simple correlation coefficient , shape = [*number]
        p_value (np.ndarray) : T test p value , shape = [*number]
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """ Simple linear regression y[idx] = slope[idx] * x + intcp[idx]

        Args:
            x (np.ndarray): shape = (time,)
            y (np.ndarray): shape = (time,*number)
            x's dim0 must equal to y'dim0 !
        """
        if isinstance(x, xr.DataArray):
            x = np.array(x)
        if isinstance(y, xr.DataArray):
            y = np.array(y)
        self.slope, self.intcpt, self.corr, self.p_value = linear_reg(x, y)


class MultLinReg:
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
        if isinstance(x, xr.DataArray):
            x = np.array(x)
        if isinstance(y, xr.DataArray):
            y = np.array(y)
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
