# SACPY -- A Python Package for Statistical Analysis of Climate

**Sacpy**, a repaid Statistical Analysis tool for Climate or Meteorology data.

Author : Zilu Meng

e-mail : mzll1202@163.com

github : https://github.com/ZiluM/sacpy

gitee : https://gitee.com/zilum/sacpy

examples or document :  https://github.com/ZiluM/sacpy/tree/master/examples or https://gitee.com/zilum/sacpy/tree/master/examples

version : 0.0.8

## Why choose Sacpy?

### Quick!

For example, Sacpy is more than 60 times faster than the traditional regression analysis with Python (see **speed test**).

### Turn to climate data customization!

Compatible with commonly used meteorological calculation libraries such as numpy and xarray.


## Install

        pip install sacpy


## Example

### example1
Calculate the correlation between SST and nino3.4 index

```Python
import numpy as np
import scapy as scp
import matplotlib.pyplot as plt


# load sst
sst = scp.load_sst()['sst']
# get ssta (method=1, Remove linear trend;method=0, Minus multi-year average)
ssta = scp.get_anom(sst,method=1)
# calculate Nino3.4
Nino34 = ssta.loc[:,-5:5,190:240].mean(axis=(1,2))
# regression
linreg = scp.LinReg(np.array(Nino34),np.array(ssta))
# plot
plt.contourf(linreg.corr)
# Significance test
plt.contourf(linreg.p_value,levels=[0, 0.05, 1],zorder=1,
            hatches=['..', None],colors="None",)
# save
plt.savefig("./nino34.png")

```
Result(For a detailed drawing process, see **example**):

![](https://raw.githubusercontent.com/ZiluM/sacpy/master/pic/nino34.png)

### example2

multiple linear regression on Nino3.4 IOD Index and ssta pattern

```Python
import numpy as np
import scapy as scp
import matplotlib.pyplot as plt


# load sst
sst = scp.load_sst()['sst']
# get ssta (method=1, Remove linear trend;method=0, Minus multi-year average)
ssta = scp.get_anom(sst,method=1)
# calculate Nino3.4
Nino34 = ssta.loc[:,-5:5,190:240].mean(axis=(1,2))
# calculate IODIdex
IODW = ssta.loc[:,-10:10,50:70].mean(axis=(1,2))
IODE = ssta.loc[:,-10:0,90:110].mean(axis=(1,2))
IODI = +IODW - IODE
# get x
X = np.vstack([np.array(Nino34),np.array(IODI)]).T
# multiple linear regression
MLR = scp.MultLinReg(X,np.array(ssta))
# plot IOD's effect
plt.contourf(MLR.slope[1])
# Significance test
plt.contourf(MLR.pv_i[1],levels=[0, 0.1, 1],zorder=1,
            hatches=['..', None],colors="None",)
plt.savefig("../pic/MLR.png")
```
Result(For a detailed drawing process, see **example**):

![](https://raw.githubusercontent.com/ZiluM/sacpy/master/pic/MLR.png)

## Speed 

As a comparison, we use the  **corr**  function in the xarray library, **corrcoef** function in numpy library and **for-loop**. The time required to calculate the correlation coefficient between SSTA and nino3.4 for 50 times is shown in the figure below.

It can be seen that we are five times faster than xarray.corr, 60 times faster than forloop and 200 times faster than numpy.corrcoef.

Moreover, xarray and numpy can not return the **p value**. We can simply check the pvalue attribute of sacpy to get the p value.

All in all, if we want to get p-value and correlation or slope, Sacpy is 60 times faster than before.

![](https://raw.githubusercontent.com/ZiluM/sacpy/master/pic/speed_test_00.png)



## Acknowledgements


Thank Prof. Feng Zhu (NUIST,https://github.com/fzhu2e) for his guidance of this project