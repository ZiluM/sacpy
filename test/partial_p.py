import numpy as np
import scapy as scp
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import datasets
import pandas as pd
# load sst
sst = scp.load_sst()['sst']
ssta = scp.get_anom(sst,method=1)

# calculate Nino3.4
Nino34 = ssta.loc[:,-5:5,190:240].mean(axis=(1,2))

IODW = ssta.loc[:,-10:10,50:70].mean(axis=(1,2))
IODE = ssta.loc[:,-10:10,90:110].mean(axis=(1,2))
IODI = IODW - IODE

sst_m = ssta.mean(axis=(1,2))
y = np.array(sst_m)[:,np.newaxis]

X = np.vstack([np.array(Nino34),np.array(IODI)]).T

X1 = sm.add_constant(X)
# X = sm.add_constant(X)
m0 = sm.OLS(y,X1).fit()
mlr = scp.MultLinReg(X,y)
# print(m0.summary())
print(mlr.pv_i)