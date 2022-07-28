import numpy as np
import scipy.stats as sts

EPS = 1e-5


class EOF():
    """ EOF analysis of data
    """

    def __init__(self, data: np.ndarray):
        """ initiation of EOF
        Args:
            data (np.ndarray): shape (time, * space grid number)
        """
        # orginal data
        self.data = data
        # original data shape
        self.origin_shape = data.shape
        # time length
        self.tLen = data.shape[0]
        # reshape (time, space)
        self.rsp_data = data.reshape(self.tLen, -1)

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

    def solve(self):
        """ solve 

        Returns:
            _type_: _description_
        """
        # mask data
        self._mask_nan()
        # solve maksed data
        data_nN = self.data_nN
        # get (time,spcae_noNan)
        dim0_len, dim1_len = data_nN.shape
        # time > space
        if dim0_len > dim1_len:  # time > space
            dim_min = dim1_len
            # get coviance (space_noNan,space_noNan)
            cov = data_nN.T @ data_nN
            # get eigenvalues and right eigenvectors
            eign, e_vector = np.linalg.eig(cov)  # (dim_min) ; (dim_min , dim_min) [i]&[:,i]
            # trans
            e_vector = e_vector.T  # [i]&[i,:]

        else:  # space > time
            # get coviance
            dim_min = dim0_len
            # get cov, (time,time)
            cov = data_nN @ data_nN.T
            # get eigenvalues and right eigenvectors
            eign, e_vector_s = np.linalg.eig(cov)  # (dim_min) ; (dim_min , dim_min) [i]&[:,i]
            # trans
            e_vector = (data_nN.T @ e_vector_s / np.sqrt(np.abs(eign))).T[:dim_min]
        # save
        pc = e_vector @ data_nN.T  # (pattern_num, time)
        self.pc = pc  # time pc (pattern_num, time)
        self.e_vector = e_vector
        self.eign = eign
        self.dim_min = dim_min
        # get patterns
        patterns = np.zeros((dim_min, *self.rsp_data.shape[1:]))
        patterns[:, self.flag] = e_vector
        # refill Nan
        patterns[:, np.logical_not(self.flag)] = np.NAN
        # patterns = patterns.reshape((dim_min, *self.origin_shape[1:]))
        # save
        self.patterns = patterns

    def get_pc(self, scaling="std"):
        """ get pc of eof analysis

        Args:
            scaling (str, optional): scale method. Defaults to "std".

        Returns:
            pc_re : pc of eof
        """
        if scaling == "std":
            pc_re = self.pc / self.pc.std(axis=1)[..., np.newaxis]
            print(self.pc.std(axis=1).shape)
        return pc_re

    def get_pt(self, scaling="mstd"):
        """ get spatial patterns of EOF analysis

        Args:
            scaling (str, optional): sacling method. Defaults to "mstd".

        Returns:
            _type_: _description_
        """
        if scaling == "mstd":
            patterns = self.patterns * self.pc.std(axis=1)[..., np.newaxis]
        # reshape to original shape (pattern_num,*space)
        patterns = patterns.reshape((self.dim_min, *self.origin_shape[1:]))
        return patterns

    def projection(self, proj_field: np.ndarray, scaling="std"):
        """ project new field to EOF spatial pattern

        Args:
            proj_field (np.ndarray): shape (time, *space grid number)
            scaling (str, optional): _description_. Defaults to "std".

        Returns:
            pc_proj : _description_
        """
        proj_field_noNan = proj_field.reshape(proj_field.shape[0], -1)[:, self.flag]
        pc_proj = self.e_vector @ proj_field_noNan.T
        if scaling == "std":
            pc_proj = pc_proj / self.pc.std(axis=1)[..., np.newaxis]
        return pc_proj

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
        norm_evctor = (need_evctor - need_evctor.mean(axis=1)[..., np.newaxis])/ \
                    (need_evctor.std(axis=1)[..., np.newaxis])
        data_noNan_norm = (data_noNan - data_noNan.mean(axis=1)[..., np.newaxis])/ \
                    (data_noNan.std(axis=1)[..., np.newaxis])

        corr = norm_evctor @ data_noNan_norm.T / (N + 2)  # npatterns,data_time
        t_value = np.abs(corr / (EPS + np.sqrt(1 - corr**2)) * np.sqrt(N))
        p_value = sts.t.sf(t_value, df=N - 2) *2
        return corr, p_value
