qstack.regression.kernel\_utils
===============================

Functions
---------

get\_local\_kernel (arg)
~~~~~~~~~~~~~~~~~~~~~~~~

::

    Obtains a local-envronment kernel by name.

    Args:
        arg (str): the name of the kernel, in ['']  # TODO

    Returns:
        kernel (Callable[np.ndarray,np.ndarray,float -> np.ndarray]): the actual kernel function, to call as ``K = kernel(X,Y,gamma)``

    .. todo::
        Write the docstring

get\_global\_kernel (arg, local\_kernel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    .. todo::
        Write the docstring

get\_kernel (arg, arg2=None)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Returns the kernel function depending on the cli argument

    .. todo::
        Write the docstring

train\_test\_split\_idx (y, idx\_test=None, idx\_train=None, test\_size=defaults.test\_size, random\_state=defaults.random\_state)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Perfrom test/train data split based on random shuffling or given indices.

        If neither `idx_test` nor `idx_train` are specified, the splitting
           is done randomly using `random_state`.
        If either `idx_test` or `idx_train` is specified, the rest idx are used
           as the counterpart.
        If both `idx_test` and `idx_train` are specified, they are returned.
        * Duplicates within `idx_test` and `idx_train` are not allowed.
        * `idx_test` and `idx_train` may overlap but a warning is raised.

    Args:
        y (numpy.1darray(Nsamples)): array containing the target property of all Nsamples
        test_size (float or int): test set fraction (or number of samples)
        idx_test ([int] / numpy.1darray): list of indices for the test set (based on the sequence in X)
        idx_train ([int] / numpy.1darray): list of indices for the training set (based on the sequence in X)
        random_state (int): the seed used for random number generator (controls train/test splitting)

    Returns:
        numpy.1darray(Ntest, dtype=int) : test indices
        numpy.1darray(Ntrain, dtype=int) : train indices
        numpy.1darray(Ntest, dtype=float) : test set target property
        numpy.1darray(Ntrain, dtype=float) : train set target property

sparse\_regression\_kernel (K\_train, y\_train, sparse\_idx, eta)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Compute the sparse regression matrix and vector.

        Solution of a sparse regression problem is
        $$ \vec w = \left( \mathbf{K}_{MN} \mathbf{K}_{NM} + \eta \mathbf{1} \right) ^{-1} \mathbf{K}_{MN}\vec y $$
        where
            w: regression weights
            N: training set
            M: sparse regression set
            y: target
            K: kernel
        This function computes K_solve: $\mathbf{K}_{MN} \mathbf{K}_{NM} + \eta \mathbf{1}$
        and y_solve $\mathbf{K}_{MN}\vec y$.

    Args:
        K_train (numpy.1darray(Ntrain1,Ntrain): kernel computed on the training set.
                Ntrain1 (N in the equation) may differ from the full training set Ntrain (e.g. a subset)
        y_train (numpy.1darray(Ntrain)): array containing the target property of the full training set
        sparse_idx (numpy.1darray of int) : (M in the equation): sparse subset indices
                   wrt to the order of the full training set.
        eta (float): regularization strength for matrix inversion

    Returns:
        numpy.2darray((len(sparse), len(sparse)), dtype=float) : matrix to be inverted
        numpy.1darray((len(sparse)), dtype=float) : vector of the constant terms

Classes
-------

ParseKwargs
~~~~~~~~~~~

(No docstring.)

Methods
:::::::

\_\_call\_\_ (self, \_parser, namespace, values, \_option\_string=None)
.......................................................................

(No docstring.)

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
