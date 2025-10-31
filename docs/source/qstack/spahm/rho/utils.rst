qstack.spahm.rho.utils
======================

Functions
---------

get\_chsp (fname, n)
~~~~~~~~~~~~~~~~~~~~

(No docstring.)

load\_mols (xyzlist, charge, spin, basis, printlevel=0, units='ANG', ecp=None, progress=False, srcdir=None)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(No docstring.)

mols\_guess (mols, xyzlist, guess, xc=defaults.xc, spin=None, readdm=None, printlevel=0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(No docstring.)

dm\_open\_mod (dm, omod)
~~~~~~~~~~~~~~~~~~~~~~~~

(No docstring.)

get\_xyzlist (xyzlistfile)
~~~~~~~~~~~~~~~~~~~~~~~~~~

(No docstring.)

check\_data\_struct (fin, local=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(No docstring.)

load\_reps (f\_in, from\_list=True, srcdir=None, with\_labels=False, local=True, sum\_local=False, printlevel=0, progress=False, file\_format=None)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    A function to load representations from txt-list/npy files.
        Args:
            - f_in: the name of the input file
            - from_list(bool): if the input file is a txt-file containing a list of paths to the representations
            - srcdir(str) : the path prefix to be at the begining of each file in `f_in`, defaults to cwd
            - with_label(bool): saves a list of tuple (filename, representation)
            - local(bool): if the representations is local
            - sum_local(bool): if local=True then sums the local components
            - printlevel(int): level of verbosity
            - progress(bool): if True shows progress-bar
            - file_format(dict): (for "experienced users" only) structure of the input data, defaults to structure auto determination
        Return:
            np.array with shape (N,M) where N number of representations M dimmensionality of the representation
            OR tuple ([N],np.array(N,M)) containing list of labels and np.array of representations

regroup\_symbols (file\_list, print\_level=0, trim\_reps=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(No docstring.)

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
