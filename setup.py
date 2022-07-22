from setuptools import setup

setup(name="sacpy",
      version="0.0.1",
      description="A repaid Statistical Analysis tool for Climate or Meteorology data.",
      py_modules={"sacpy"},
      package_dir={'': "sacpy"},
      author="Zilu Meng",
      long_description=open("./READE.md").read() + "\n\n" + open("./CHANGELOG.md").read(),
      author_email="mzll1202@163.com",
      install_requires=[""],
      license="MIT")