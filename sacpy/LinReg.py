from .linger_cal import linear_reg, multi_linreg
import numpy as np


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

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Args:
            x (np.ndarray): shape = (time, N) ; N is Number of factors
            y (np.ndarray): shape = (time,*number)
            x's dim0 must equal to y'dim0 !
        """
        self.slope, self.intcpt, self.R, self.pv_all, self.pv_i = multi_linreg(x, y)
