import numpy as np
import scipy.stats as sts

EPS = 1e-10


def linear_reg(x: np.ndarray, y: np.ndarray):
    """ Simple linear regression y[idx] = slope[idx] * x + intcp[idx]
    Args:
        x (np.ndarray): shape = (time,)
        y (np.ndarray): shape = (time,*number)
        x's dim0 must equal to y'dim0 !

    Returns:
        slope (np.ndarray): Slope, shape = [*number]
        intcpt (np.ndarray) : intercept , shape = [*number]
        corr (np.ndarray) : Simple correlation coefficient , shape = [*number]
        p_value (np.ndarray) : T test p value , shape = [*number]
    """
    # check shape
    Num0 = x.shape[0]
    Num1 = y.shape[0]
    # Initial Y shape
    org_sp_Y = y.shape
    # no equal, raise error
    if Num0 != Num1:
        raise (ValueError("x.shape[0] no equal to y.shape[0] , dim0 is wrong"))
    # reshape y to (time,number)
    y_rs = y.reshape(Num1, -1)
    # cal anomaly
    xa = x - x.mean(axis=0)
    ya = y_rs - y_rs.mean(axis=0)
    # cal std
    y_std = ya.std(axis=0)
    x_std = xa.std(axis=0)
    # cal covariance
    covar = ya.T @ xa / Num0
    # cal corr
    corr = covar / (y_std + EPS) / (x_std + EPS)
    # cal slope
    slope = corr * y_std / (x_std + EPS)
    # cal intcpt
    intcpt = y_rs.mean(axis=0) - slope * x.mean(axis=0)
    # cal t-valpue
    t = corr / (np.sqrt(1 - corr**2) + EPS) * np.sqrt(Num0 - 2)
    p_value = sts.t.sf(t, df=Num0 - 2)
    # transform t value
    pv_cp = np.copy(p_value)
    p_value[pv_cp >= 0.5] = (1 - p_value[pv_cp >= 0.5]) * 2
    p_value[pv_cp < 0.5] = (p_value[pv_cp < 0.5]) * 2
    # reshape
    slope, intcpt, corr, p_value = list(map(lambda inar: inar.reshape(org_sp_Y[1:]), [slope, intcpt, corr, p_value]))
    # return result
    return slope, intcpt, corr, p_value


def multi_linreg(x: np.ndarray, y: np.ndarray):
    """ multiple linear regression :: y[idx] = slope[idx,0] * x[0] + slope[idx, 1] * x[1] + ... + intcp[idx]

    Args:
        x (np.ndarray): shape = (time, N) ; N is Number of factors
        y (np.ndarray): shape = (time,*number)
        x's dim0 must equal to y'dim0 !
    Returns:
        slope (np.ndarray): Slope, shape = [N,*number]
        intcpt (np.ndarray) : intercept , shape = [*number]
        R (np.ndarray) : multiple correlation coefficient , shape = [*number]
        pv_all(np.ndarray) : F test p value , shape = [*number]
        pv_i(np.ndarray) : F test p value of every infact, shape = [N, *number]
    """
    # save shape
    Num0 = x.shape[0]
    Num1 = y.shape[0]
    # number of factors
    Numf = x.shape[1]
    # y initial shape
    org_sp_Y = y.shape
    # raise error
    if Num0 != Num1:
        raise (ValueError("x.shape[0] no equal to y.shape[0] , dim0 is wrong"))
    # reshape y
    y_rs = y.reshape(Num1, -1)
    # cal mean
    x_mean = x.mean(axis=0)
    y_mean = y_rs.mean(axis=0)
    # get anomaly
    xa = x - x_mean
    ya = y_rs - y_mean
    # cal covar
    covar_xx = xa.T @ xa / Num0
    covar_yx = xa.T @ ya / Num0
    # get slope
    slope = np.linalg.solve(covar_xx, covar_yx)  # (Numf , Num space)
    # get intcpt
    intcpt = y_mean - slope.T @ x_mean
    # get y_pre
    y_pre = xa @ slope
    # Total variance
    SSyy = Num0 * y_rs.var(axis=0)
    # Regression interpretation variance
    U = Num0 * y_pre.var(axis=0)
    # residual
    Q = SSyy - U
    # correlation coefficient
    R = np.sqrt(U / SSyy)  # (Num space)
    # F value of total regression equation
    F = U / Numf / (Q / (Num0 - Numf - 1))
    # F value of Infactor
    C_mx = np.diagonal(np.linalg.inv(covar_xx * Num0))
    Q_mx = slope**2 / C_mx[..., np.newaxis]  # (Numf,Num space)
    F_i = Q_mx / (Q / (Num0 - Numf - 1))  # (Numf,Num space)
    # p value
    pv_all = 1 - sts.f.cdf(F, Num0, Num0 - Numf - 1)  # (Num space)
    pv_i = 1 - sts.f.cdf(F_i, 1, Num0 - Numf - 1)  # (Numf,Num space)
    # reshape
    pv_all, R, intcpt = list(map(lambda inar: inar.reshape(org_sp_Y[1:]), [pv_all, R, intcpt]))
    slope, pv_i = list(map(lambda inar: inar.reshape([Numf, *org_sp_Y[1:]]), [slope, pv_i]))
    # return
    return slope, intcpt, R, pv_all, pv_i


def multi_corr(x: np.ndarray, y: np.ndarray):
    # save shape
    Num0 = x.shape[0]
    Num1 = y.shape[0]
    # number of factors
    Numf = x.shape[1]
    # y initial shape
    org_sp_Y = y.shape
    # raise error
    if Num0 != Num1:
        raise (ValueError("x.shape[0] no equal to y.shape[0] , dim0 is wrong"))
    # reshape y
    y_rs = y.reshape(Num1, -1)
    # Num space
    Nums = y_rs.shape[1]
    # cal mean
    x_mean = x.mean(axis=0)
    y_mean = y_rs.mean(axis=0)
    # get anomaly
    xa = x - x_mean
    ya = y_rs - y_mean
    # cal std
    x_std = xa.std(axis=0)
    y_std = ya.std(axis=0)
    # normalization
    x_norm = xa / x_std
    y_norm = ya / y_std
    # before correlation
    x_norm_b = np.repeat(x_norm[..., np.newaxis], axis=2, repeats=Nums)  # (t,Nf,Ns)
    y_norm_b = y_norm[:, np.newaxis, :]  # (t , 1 ,Ns)
    conc_xy = np.concatenate([x_norm_b, y_norm_b], axis=1)  # (t,Nf+1,Ns)
    corr_matrix = np.transpose(conc_xy, axes=[2, 1, 0]) @ np.transpose(conc_xy, axes=[2, 0, 1]) / Num0  # (Ns,Nf+1,Nf+1)
    corr_matrix = corr_matrix.reshape([*org_sp_Y[1:], Numf + 1, Numf + 1])
    return corr_matrix


def partial_corr(x: np.ndarray, y: np.ndarray, indx: int, mul_corr=None):
    """ calculate partial correlation

    Args:
        x (np.ndarray): shape = (time, N) ; N is Number of factors
        y (np.ndarray): shape = (time,*number)
        mul_corr (np.ndarray, optional): _description_. Defaults to None.
        idnx (np.ndarray) : correlation between y and x[:,indx], 
                Exclude the influence of other factors
        x's dim0 must equal to y'dim0 !
    Returns:
        np.ndarray: correlation between y and x[:,idx], Exclude the influence of other factors
    """
    if mul_corr is None:
        corr_matrix = multi_corr(x, y)
    Numf = corr_matrix.shape[1] - 1
    corr_matrix = corr_matrix.reshape((-1, Numf + 1, Numf + 1))
    Rsyy = np.linalg.det(corr_matrix[:, :-1, :-1])
    Rsii = np.linalg.det(np.delete(np.delete(corr_matrix, indx, 1), indx, 2))
    Rsyi = (-1)**(Numf + indx) * np.linalg.det(np.delete(np.delete(corr_matrix, indx, 1), Numf, 2))
    return -Rsyi / np.sqrt(Rsyy * Rsii)
