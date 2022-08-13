import numpy as np
import xarray as xr
from .Util import _correct_type

EPS = 1e-5


class SVD():
    name = "SVD"
    """ SVD analysis of data
    """

    def __init__(self, data1: np.ndarray, data2: np.ndarray, complex=False):
        """ initiation of EOF
        Args:
            data1 and data2 (np.ndarray): shape (time, * space grid number)
        """
        # original data
        if complex:
            dtype_np = np.complex128
        else:
            dtype_np = np.float64
        self.dtype_np = dtype_np
        self._pc_std_save = 0
        # if isinstance(data1, xr.DataArray):
        #     data1 = np.array(data1, dtype=dtype_np)
        data1 = _correct_type(data1, dtype=dtype_np)
        # if isinstance(data2, xr.DataArray):
        #     data2 = np.array(data2, dtype=dtype_np)
        data2 = _correct_type(data2, dtype=dtype_np)
        self.origin_shape1 = data1.shape
        self.origin_shape2 = data2.shape
        # time length
        if data1.shape[0] != data2.shape[0]:
            raise ValueError("data1.time [%s] != data2.time [%s]" % (data1.shape[0], data2.shape[0]))
        self.tLen = data1.shape[0]
        # reshape (time, space)
        self.rsp_data1 = data1.reshape(self.tLen, -1)
        self.rsp_data2 = data2.reshape(self.tLen, -1)

    def _mask_nan(self, data):
        """ mask Nan or drop out the Nan
        """
        # get not Nan flag
        flag = np.isnan(data).sum(axis=0) == 0
        # save to self.flag
        self.flag = flag
        # get data without Nan
        data_nN = data[:, flag]  # (time, space_noNan)
        return data_nN, flag

    def solve(self):
        """ solve the SVD
        """
        # mask data
        data1_noNan, flag1 = self._mask_nan(self.rsp_data1)
        data2_noNan, flag2 = self._mask_nan(self.rsp_data2)
        self._data1_noNan, self._data2_noNan = data1_noNan, data2_noNan
        self.flag1, self.flag2 = flag1, flag2
        # get covariance
        Covan = data1_noNan.T @ data2_noNan
        # svd solver
        U_matrix, S, V_matrix = np.linalg.svd(Covan)
        # save
        self._U = U_matrix
        self._V = V_matrix
        self.eign = S

    def get_eign(self):
        """ return the eign of coviance"""
        return self.eign

    def get_varperc(self, npt):
        """ return percentile of variance eign"""
        var_perc = self.eign[:npt] / np.sum(self.eign)
        return var_perc

    def get_pc(self, npt, norm="std"):
        pc_left = self._U[:, :npt].T @ self._data1_noNan.T
        pc_right = self._V[:, :npt].T @ self._data2_noNan.T
        pc_left, pc_right = np.real(pc_left), np.real(pc_right)
        if norm == "std":
            self.pc_left_std = pc_left.std(axis=1)[:, np.newaxis]
            self.pc_right_std = pc_right.std(axis=1)[:, np.newaxis]
            pc_left /= self.pc_left_std
            pc_right /= self.pc_right_std
        else:
            self.pc_left_std = pc_left.std(axis=1)[:, np.newaxis]
            self.pc_right_std = pc_right.std(axis=1)[:, np.newaxis]
        self._pc_std_save = npt
        return pc_left, pc_right

    def get_pt(self, npt, norm="std"):
        #
        patterns_left = np.zeros((npt, *self.rsp_data1.shape[1:]), dtype=self.dtype_np)
        patterns_right = np.zeros((npt, *self.rsp_data2.shape[1:]), dtype=self.dtype_np)
        if norm == "std":
            if npt > self._pc_std_save:
                self.get_pc(npt=npt)
            else:
                pass
            patterns_left *= self.pc_left_std[:npt]
            patterns_right *= self.pc_right_std[:npt]
        else:
            pass
        # 
        patterns_left[:, self.flag1] = self._U[:, :npt].T
        patterns_left[:, np.logical_not(self.flag1)] = np.NAN
        patterns_left = patterns_left.reshape((npt, *self.origin_shape1[1:]))
        #

        patterns_right[:, self.flag2] = self._V[:npt]
        patterns_right[:, np.logical_not(self.flag2)] = np.NAN
        patterns_right = patterns_right.reshape((npt, *self.origin_shape2[1:]))

        return patterns_left, patterns_right
