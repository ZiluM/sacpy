import numpy as np
import scipy.stats as sts
import xarray as xr
from .Util import _correct_type0, _correct_type, gen_dataarray


class STMV:
    """ Significance T-Test of mean value
    """
    name = "significance-test-of-mean-value"

    def __init__(self, data1, data2=None, wrap=True, *param, **kwargs):
        """ 
        Args:
            data1 (np.ndarray, xr.Dataarray): input data1 (time, *nspace)
            data2 (np.ndarray, xr.Dataarray, optional): input data1 (time, *nspace)
            wrap (bool, optional): if data are wrapped to original xarray coords. Defaults to True.
        """
        self.data1 = data1
        self.data2 = data2
        data1, dims1, coords1 = _correct_type0(data1)
        data2, dims2, coords2 = _correct_type0(data2)
        if data2 is None:
            mean, p_value = one_mean_test(data1, *param, **kwargs)
        elif data2 is not None:
            mean, p_value = two_mean_test(data1, data2, *param, **kwargs)
        if wrap is True:
            if dims1 is not None and coords1 is not None:
                coords_m1 = {dims1[i]: coords1[dims1[i]] for i in range(1, len(dims1))}
                mean, p_value = map(lambda data: gen_dataarray(data, coords_m1), [mean, p_value])
        self.mean, self.p_value = mean, p_value


    def get_original_data(self):
        if self.data2 is not None:
            return self.data1, self.data2
        else:
            return self.data1





def one_mean_test(data: np.ndarray, expected_mean=None, *param) -> np.ndarray:
    """ one sample t test
    Args:
        data (np.ndarray): test data
        expected_mean (np.ndarray, optional): expected mean. Defaults to None.
        *param: parameter transports to sts.ttest_1samp

    Raises:
        TypeError: data's type should be np.ndarray
        ValueError: expected_mean shape can match data

    Returns:
        origin_mean,pvalue : np.ndarray: origin_mean and res.value
    """
    if isinstance(data, (xr.DataArray, np.ndarray)):
        data = np.array(data).copy()
    else:
        raise TypeError("Type should be np.ndarray or xr.Dataarray, rather than %s" % type(data))
    if expected_mean is None:
        expected_mean = np.zeros(data.shape[1:])
    else:
        if expected_mean.shape != data.shape[1:]:
            raise ValueError("expected_mean shape can match data")
    origin_mean = data.mean(axis=0)
    res = sts.ttest_1samp(data, expected_mean, axis=0, *param)
    return origin_mean, res.pvalue


def two_mean_test(data1: np.ndarray, data2: np.ndarray, *param) -> np.ndarray:
    """two samples t test

    Args:
        data1 (np.ndarray): test data1
        data2 (np.ndarray): test data2

    Raises:
        TypeError: Type should be np.ndarray

    Returns:
        mean_diff, pvalue np.ndarray: origin_mean difference and res.value
    """
    if isinstance(data1, (xr.DataArray, np.ndarray)) and isinstance(data2, (xr.DataArray, np.ndarray)):
        data1 = np.array(data1).copy()
        data2 = np.array(data2).copy()
    else:
        raise TypeError("Type should be np.ndarray or xr.Dataarray, rather than %s" % type(data1))
    mean_diff = data1.mean(axis=0) - data2.mean(axis=0)
    res = sts.ttest_ind(data1, data2, axis=0, *param)
    return mean_diff, res.pvalue
