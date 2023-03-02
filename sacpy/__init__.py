from .version import __version__
from .LinReg import LinReg, MultLinReg
from .XrTools import get_anom, spec_moth_dat, spec_moth_yrmean
from .Util import convert_lon, reverse_lon, sig_scatter
from .load_sst import load_sst,load_10mwind
from .EOF import EOF
from .linger_cal import linear_reg, multi_linreg, multi_corr, partial_corr
from .SigTest import one_mean_test, two_mean_test
from .SVD import SVD


__author__ = "Zilu Meng"
__email__ = "mzll1202@163.com"