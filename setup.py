from setuptools import setup, find_packages

with open("README.md","r") as fh:
      long_des1 = fh.read()
with open("CHANGELOG.md","r") as fh1:
      long_des2 = fh1.read()
long_des = long_des1 + "\n\n" + long_des2
# print(long_des)


print(find_packages())
setup(name="sacpy",
      version="0.0.20",
      url="https://github.com/ZiluM/sacpy",
      description="A repaid Statistical Analysis tool for Climate or Meteorology data.",
      py_modules=["sacpy"],
      include_package_data=True,
      keywords="meteorology data statistic climate",
      author="Zilu Meng",
      author_email="zilumeng@uw.edu",
      install_requires=["numpy", "scipy", "xarray"],
      package_data={'sacpy': ['data/example/HadISST_sst_5x5.nc','data/example/NCEP_wind10m_5x5.nc']},
      packages=find_packages(),
      long_description=long_des,
      long_description_content_type="text/markdown",
      license="MIT")



