import numpy as np
import scipy.stats as sts
import xarray as xr
from .LinReg import LinReg
import time

EPS = 1e-5


class EOF:
    """ EOF analysis of data
    """
    name = "EOF"

    def __init__(self, data: np.ndarray, weights=None):
        """ initiation of EOF
        Args:
            data (np.ndarray): shape (time, * space grid number)
            weights : shape (* space grid number , 
                        or can be broadcast to space grid number)
        """
        # original data
        if isinstance(data, xr.DataArray):
            data = np.array(data)
        if weights is None:
            self.data = np.copy(data)
        else:
            self.data = np.copy(data) * weights
        # original data shape
        self.origin_shape = data.shape
        # time length
        self.tLen = data.shape[0]
        # reshape (time, space)
        self.rsp_data = data.reshape(self.tLen, -1)
        self.pc = None
        self.got_pc_num = 0

    def _mask_nan(self):
        """ mask Nan or drop out the Nan
        """
        # get data0
        data0 = self.rsp_data
        # get not Nan flag
        flag = np.isnan(data0).sum(axis=0) == 0
        # save to self.flag
        self.flag = flag
        # get data without Nan
        data_nN = data0[:, flag]  # (time, space_noNan)
        # save
        self.data_nN = data_nN

    # def _mask_extra_data(self,data):
    #     flag = np.isnan(data0).sum(axis=0) == 0

    def solve(self, method="eig",st=False):
        """ solve the EOF
        """
        if method not in ['eig', 'svd']:
            raise ValueError(f"method must be 'eig' or 'svd', not {method}")
        # mask data
        self._mask_nan()
        # solve maksed data
        data_nN = self.data_nN
        # get (time,spcae_noNan)
        dim0_len, dim1_len = data_nN.shape
        # print("Start EOF")
        # ================================= EOF process by SVD===============================
        if st is True:
            print(f"=====EOF Start at {time.time()}======")
        if method == "svd":
            Xor = 1 / np.sqrt(dim0_len - 1) * data_nN
            U, Sigma, VT = np.linalg.svd(Xor)
            e_vector = VT
            eign = Sigma
            dim_min = np.min([dim0_len, dim1_len])
        # ================================= EOF process end===============================
        elif method == "eign":
            if dim0_len > dim1_len:  # time > space
                print(1)
                dim_min = dim1_len
                # get coviance (space_noNan,space_noNan)
                cov = data_nN.T @ data_nN
                # get eigenvalues and right eigenvectors
                eign, e_vector = np.linalg.eig(cov)  # (dim_min) ; (dim_min , dim_min) [i]&[:,i]
                # trans
                e_vector = e_vector.T  # [i]&[i,:]

            else:  # space > time
                print(2)
                # get coviance
                dim_min = dim0_len
                # get cov, (time,time)
                cov = data_nN @ data_nN.T
                # get eigenvalues and right eigenvectors
                eign, e_vector_s = np.linalg.eig(cov)  # (dim_min) ; (dim_min , dim_min) [i]&[:,i]
                # trans
                e_vector = (data_nN.T @ e_vector_s / np.sqrt(np.abs(eign))).T[:dim_min]
        if st is True:
            print(f"=====EOF End at  {time.time()}======")
        # save
        # print("EOF End")
        self.e_vector = e_vector
        self.eign = eign
        self.dim_min = dim_min
        # get patterns
        patterns = np.zeros((dim_min, *self.rsp_data.shape[1:]))
        patterns[:, self.flag] = e_vector[:dim_min]
        # refill Nan
        patterns[:, np.logical_not(self.flag)] = np.NAN
        # patterns = patterns.reshape((dim_min, *self.origin_shape[1:]))
        # save
        self.patterns = patterns

    def get_eign(self):
        """ get eign of each pattern
        """
        return self.eign

    def get_varperc(self, npt=None):
        """ return variance percential

        Args:
            npt (int, optional): n patterns to get. Defaults to None.

        Returns:
            variace percentile (np.ndarray): variance percentile (npt,)
        """
        if npt is None:
            npt = self.dim_min
        var_perc = self.eign[:npt] / np.sum(self.eign)
        return var_perc

    def get_pc(self, npt=None, scaling="std"):
        """ get pc of eof analysis

        Args:
            scaling (str, optional): scale method. None, 'std','DSE' and 'MSE'. Defaults to "std".

        Returns:
            pc_re : pc of eof (pattern_num, time)
        """

        if npt is None:
            npt = self.dim_min
        pc = self.e_vector[:npt] @ self.data_nN.T  # (pattern_num, time)
        # self.pc = pc
        self.pc_std = pc[:npt].std(axis=1)[..., np.newaxis]
        if scaling == "std":
            pc_re = pc[:npt] / self.pc_std
        elif scaling == "DSE":
            pc_re = pc[:npt] / np.sqrt(self.eign[:npt, ..., np.newaxis])
        elif scaling == "MSE":
            pc_re = pc[:npt] * np.sqrt(self.eign[:npt, ..., np.newaxis])
        elif scaling is None:
            pc_re = pc
        else:
            raise ValueError(f"invalid PC scaling option: '{scaling}', Must be None, 'std','DSE' and 'MSE' ")
        self.pc = pc_re
        self.got_pc_num = npt
        return pc_re

    def get_pt(self, npt=None, scaling="mstd"):
        """ get spatial patterns of EOF analysis

        Args:
            scaling (str, optional): sacling method. None, 'std','DSE' and 'MSE'. Defaults to "mstd".

        Returns:
            patterns : (pattern_num, * space grid number)
        """
        if npt is None:
            npt = self.dim_min
        if self.pc is None or self.got_pc_num < npt:
            self.get_pc(npt=npt)
        if scaling == "mstd":
            # print(self.patterns[:npt].shape)
            # print(self.pc_std[:npt, ..., np.newaxis].shape)
            patterns = self.patterns[:npt] * self.pc_std[:npt]
        elif scaling == "MSE":
            patterns = self.patterns[:npt] * self.eign[:npt, ..., np.newaxis]
        elif scaling == "DSE":
            patterns = self.patterns[:npt] / self.eign[:npt, ..., np.newaxis]
        else:
            patterns = self.patterns[:npt]
        # reshape to original shape (pattern_num,*space)
        patterns = patterns.reshape((npt, *self.origin_shape[1:]))
        return patterns

    def correlation_map(self, npt):
        """ Get correlation map

        Args:
            npt 

        Returns:
            correlation map (npt, ...)
        """
        pcs = self.get_pc(npt=npt)
        corr_map_ls = []
        for i in range(npt):
            pc = pcs[i]
            Lin_map = LinReg(pc, self.data).corr
            corr_map_ls.append(Lin_map[np.newaxis, ...])
        corr_maps = np.concatenate(corr_map_ls, axis=0)
        return corr_maps

    def projection(self, proj_field: np.ndarray, npt=None, scaling="std"):
        """ project new field to EOF spatial pattern

        Args:
            proj_field (np.ndarray): shape (time, *space grid number)
            scaling (str, optional): _description_. Defaults to "std".

        Returns:
            pc_proj : _description_
        """
        if npt is None:
            npt = self.dim_min
        proj_field_noNan = proj_field.reshape(proj_field.shape[0], -1)[:, self.flag]
        pc_proj = self.e_vector @ proj_field_noNan.T
        if self.pc is None or self.got_pc_num < npt:
            self.get_pc(npt)
        if scaling == "std":
            pc_proj = pc_proj / self.pc.std(axis=1)[..., np.newaxis]
        return pc_proj

    def decoder(self):
        pass

    def pattern_corr(self, data: np.ndarray, npt=None):
        """  calculate pattern correlation of extra data and patterns

        Args:
            data (np.ndarray): shape (time, * space number)
            npt (int, optional): number of spatial patterns . Defaults to None.

        Returns:
            corr , p_value: corr and p_value shape (npt,data_time)
        """
        if npt is None:
            npt = self.dim_min
        # mask data
        data_noNan = data.reshape(data.shape[0], -1)[:, self.flag]
        # get need e_vector
        need_evctor = self.e_vector[:npt]
        # free degree
        N = need_evctor.shape[1] - 2
        # normalize data
        norm_evctor = (need_evctor - need_evctor.mean(axis=1)[..., np.newaxis]) / \
                      (need_evctor.std(axis=1)[..., np.newaxis])
        data_noNan_norm = (data_noNan - data_noNan.mean(axis=1)[..., np.newaxis]) / \
                          (data_noNan.std(axis=1)[..., np.newaxis])

        corr = norm_evctor @ data_noNan_norm.T / (N + 2)  # npatterns,data_time
        t_value = np.abs(corr / (EPS + np.sqrt(1 - corr**2)) * np.sqrt(N))
        p_value = sts.t.sf(t_value, df=N - 2) * 2
        return corr, p_value
