qstack.equio
============

Functions
---------

\_get\_mrange (l)
~~~~~~~~~~~~~~~~~

(No docstring.)

\_get\_llist (q, mol)
~~~~~~~~~~~~~~~~~~~~~

::

    Args:
        q (int): Atomic number.
        mol (pyscf Mole): pyscf Mole object.

    Returns:
        A list

\_get\_tsize (tensor)
~~~~~~~~~~~~~~~~~~~~~

::

    Computes the size of a tensor.

    Args:
        tensor (metatensor TensorMap): Tensor.

    Returns:
        The size of the tensor as an integer.

\_labels\_to\_array (labels)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Represents a set of metatensor labels as an array of the labels, using custom dtypes

    Args:
        labels (metatensor Labels): Labels

    Returns:
        labels (numpy ndarray[ndim=1, structured dtype]): the same labels

vector\_to\_tensormap (mol, c)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Transform a vector into a tensor map. Used by :py:func:`array_to_tensormap`.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        v (numpy ndarray): Vector.

    Returns:
        A metatensor tensor map.

tensormap\_to\_vector (mol, tensor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Transform a tensor map into a vector. :py:func:`Used by tensormap_to_array`.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        tensor (metatensor TensorMap): Tensor.

    Returns:
        A numpy ndarray (vector).

matrix\_to\_tensormap (mol, dm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Transform a matrix into a tensor map. Used by :py:func:`array_to_tensormap`.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        v (numpy ndarray): Matrix.

    Returns:
        A metatensor tensor map.

tensormap\_to\_matrix (mol, tensor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Transform a tensor map into a matrix. Used by :py:func:`tensormap_to_array`.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        tensor (metatensor TensorMap): Tensor.

    Returns:
        A numpy ndarray (matrix).

array\_to\_tensormap (mol, v)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Transform an array into a tensor map.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        v (numpy ndarray): Array. It can be a vector or a matrix.

    Returns:
        A metatensor tensor map.

tensormap\_to\_array (mol, tensor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Transform a tensor map into an array.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        tensor (metatensor TensorMap): Tensor.

    Returns:
        A numpy ndarray. Matrix or vector, depending on the key names of the tensor.

join (tensors)
~~~~~~~~~~~~~~

::

    Merge two or more tensors with the same label names avoiding information duplictaion.

    Args:
        tensors (list): List of metatensor TensorMap.

    Returns:
        A metatensor TensorMap containing the information of all the input tensors.

split (tensor)
~~~~~~~~~~~~~~

::

    Split a tensor based on the molecule information stored within the input TensorMap.

    Args:
        tensor (metatensor TensorMap): Tensor containing several molecules.

    Returns:
        N metatensor TensorMap, where N is equal to the total number of diferent molecules stored within the input TensorMap.

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
