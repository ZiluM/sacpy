# SACPY -- A Python Package for Statistical Analysis of Climate

**Sacpy**, a repaid Statistical Analysis tool for Climate or Meteorology data.

Author : Zilu Meng

e-mail : mzll1202@163.com

github : https://github.com/ZiluM/sacpy

gitee : https://gitee.com/zilum/sacpy

pypi : https://pypi.org/project/sacpy/

examples or document :  https://github.com/ZiluM/sacpy/tree/master/examples or https://gitee.com/zilum/sacpy/tree/master/examples

version : 0.0.11

## Why choose Sacpy?

### Quick!

For example, Sacpy is more than 60 times faster than the traditional regression analysis with Python (see **speed test**).

### Turn to climate data customization!

Compatible with commonly used meteorological calculation libraries such as numpy and xarray.


## Install

You can use pip to install.

        pip install sacpy

Or you can visit https://gitee.com/zilum/sacpy/tree/master/dist to download **.whl file**, then

        pip install .whl_file

## Speed 

As a comparison, we use the  **corr**  function in the xarray library, **corrcoef** function in numpy library, cdist in scipy, apply_func in xarray  and **for-loop**. The time required to calculate the correlation coefficient between SSTA and nino3.4 for 50 times is shown in the figure below.

It can be seen that we are four times faster than scipy cdist, five times faster than xarray.corr, 60 times faster than forloop, 110 times faster than xr.apply_func and 200 times faster than numpy.corrcoef.

Moreover, xarray and numpy can not return the **p value**. We can simply check the pvalue attribute of sacpy to get the p value.

All in all, if we want to get p-value and correlation or slope, we only to choose **Sacpy is 60 times faster than before**.

![](https://raw.githubusercontent.com/ZiluM/sacpy/master/pic/speed_test_00.png)


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

### example3

What effect will ENSO have on the sea surface temperature in the next summer?

```Python
import numpy as np
import sacpy as scp
import matplotlib.pyplot as plt
import xarray as xr

# load sst
sst = scp.load_sst()['sst']
ssta = scp.get_anom(sst)

# calculate Nino3.4
Nino34 = ssta.loc[:,-5:5,190:240].mean(axis=(1,2))

# get DJF mean Nino3.4
DJF_nino34 = scp.XrTools.spec_moth_yrmean(Nino34,[12,1,2])

# get JJA mean ssta
JJA_ssta = scp.XrTools.spec_moth_yrmean(ssta, [6,7,8])

# regression
reg = scp.LinReg(np.array(DJF_nino34)[:-1], np.array(JJA_ssta))
# plot
plt.contourf(reg.corr)
# Significance test
plt.contourf(reg.p_value,levels=[0, 0.05, 1],zorder=1,
            hatches=['..', None],colors="None",)
# save
plt.savefig("./ENSO_Next_year_JJA.png",dpi=300)
```

![](https://raw.githubusercontent.com/ZiluM/sacpy/master/pic/ENSO_Next_year_JJA.png)

Same as **Indian Ocean Capacitor Effect on Indo–Western Pacific Climate during the Summer following El Niño** (Xie et al.), the El Nino will lead to Indian ocean warming in next year JJA.

### example4

EOF analysis

```Python
import sacpy as scp
import numpy as np
import matplotlib.pyplot as plt
# get data
sst = scp.load_sst()["sst"].loc[:, -20:30, 150:275]
ssta = scp.get_anom(sst)
# EOF
eof = scp.EOF(np.array(ssta))
eof.solve()
# get spartial pattern and pc
pc = eof.get_pc(npt=2)
pt = eof.get_pt(npt=2)
# plot
plt.figure(figsize=[12,10])
plt.subplot(221)
plt.contourf(pt[0,:,:])
plt.colorbar()
plt.subplot(222)
plt.plot(sst.time,pc[0])
plt.subplot(223)
plt.contourf(pt[1,:,:])
plt.colorbar()
plt.subplot(224)
plt.plot(sst.time,pc[1])
plt.savefig("../pic/eof_ana.png",dpi=300)
```

![](https://raw.githubusercontent.com/ZiluM/sacpy/master/pic/eof_ana.png)




## Acknowledgements


Thank Prof. Feng Zhu (NUIST,https://github.com/fzhu2e) for his guidance of this project