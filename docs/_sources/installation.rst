Installation
===============

Quick Install
-------------

You can use pip to install directly.

.. code-block:: bash

  pip install sacpy


Install Sacpy in the Conda environment
----------------------------------------

You may skip this step if your Conda environment has been installed already.

Step 1: Download the installation script for miniconda3
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

macOS (Intel)
'''''''''''''

.. code-block:: bash

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

macOS (Apple Silicone)
'''''''''''''''''''''''

.. code-block:: bash

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

Linux
''''''
.. code-block:: bash

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

Windows
''''''''
.. code-block:: bash

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe # or Copy Link to Browser for download

Step 2: Install Miniconda3
"""""""""""""""""""""""""""

.. code-block:: bash

  chmod +x Miniconda3-latest-*.sh && ./Miniconda3-latest-*.sh # windows just click the .exe and install

During the installation, a path :code:`<base-path>` needs to be specified as the base location of the python environment.
After the installation is done, we need to add the two lines into your shell environment (e.g., :code:`~/.bashrc` or :code:`~/.zshrc`) as below to enable the :code:`conda` package manager (remember to change :code:`<base-path>` with your real location):

.. code-block:: bash

  export PATH="<base-path>/bin:$PATH"
  . <base-path>/etc/profile.d/conda.sh

Windows can add conda-bin path to $PATH automatically.

Step 3: Test your Installation
"""""""""""""""""""""""""""""""

.. code-block:: bash

  source ~/.bashrc  # assume you are using Bash shell
  which python  # should return a path under <base-path>
  which conda  # should return a path under <base-path>

Windows can skip this step, or

.. code-block:: bash

  conda activate 


Step 4: Install `Sacpy`
""""""""""""""""""""""""


Taking a clean installation as example, first let's create a new environment named :code:`cfr-env` via :code:`conda`

.. code-block:: bash

    conda create -n sacpy python=3.9
    conda activate sacpy

Then install some dependencies via :code:`conda`:

.. code-block:: bash

    conda install jupyter notebook cartopy xarray scipy numpy pandas netcdf4

Once the above dependencies have been installed, simply

.. code-block:: bash

    pip install sacpy

and you are ready to

.. code-block:: python

    import sacpy as scp

in Python.
