"""Equilibrium geometry and molecular structure I/O utilities."""

import itertools
from functools import reduce
from types import SimpleNamespace
import numpy as np
from pyscf import data
import metatensor
from qstack.tools import Cursor
from qstack.reorder import get_mrange, pyscf2gpr_l1_order
from qstack.compound import singleatom_basis_enumerator


vector_label_names = SimpleNamespace(
    tm = ['o3_lambda', 'center_type'],
    block_prop = ['radial_channel'],
    block_samp = ['atom_id'],
    block_comp = ['spherical_harmonics_m'],
    )

matrix_label_names = SimpleNamespace(
    tm = ['o3_lambda1', 'o3_lambda2', 'center_type1', 'center_type2'],
    block_prop = ['radial_channel1', 'radial_channel2'],
    block_samp = ['atom_id1', 'atom_id2'],
    block_comp = ['spherical_harmonics_m1', 'spherical_harmonics_m2'],
    )

_molid_name = 'mol_id'



def _get_llist(mol):
    """Get list of angular momentum quantum numbers for basis functions of each element of a molecule.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.

    Returns:
        dict: Dictionary with atom numbers as keys and List of angular momentum quantum numbers for each basis function as values.
    """
    return {int(q): singleatom_basis_enumerator(mol._basis[data.elements.ELEMENTS[q]])[0] for q in np.unique(mol.atom_charges())}


def _get_tsize(tensor):
    """Computes the size of a tensor.

    Args:
        tensor (metatensor.TensorMap): Tensor.

    Returns:
        int: Total size of the tensor (total number of elements).
    """
    return sum([np.prod(tensor.block(key).values.shape) for key in tensor.keys])


def _labels_to_array(labels):
    """Represents a set of metatensor labels as an array.

    Args:
        labels (metatensor.Labels): Labels object.

    Returns:
        numpy.ndarray: 1D structured array containing the same labels.
    """
    values = labels.values
    dtype = [(name, values.dtype) for name in labels.names]
    return values.view(dtype=dtype).reshape(values.shape[0])


def vector_to_tensormap(mol, c):
    """Transforms an vector into a tensor map.

    Each element of the vector corresponds to an atomic orbital of the molecule.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        c (numpy.ndarray): vector to transform.

    Returns:
        metatensor.TensorMap: Tensor map representation of the vector.
    """
    atom_charges = mol.atom_charges()

    tm_label_vals = []
    block_prop_label_vals = {}
    block_samp_label_vals = {}
    block_comp_label_vals = {}

    blocks = {}
    llists = {}

    # Create labels for TensorMap, lables for blocks, and empty blocks

    llists = _get_llist(mol)

    for q, samples_count in zip(*np.unique(atom_charges, return_counts=True), strict=True):
        llist = llists[q]
        block_samp_label_vals_q = np.where(atom_charges==q)[0].reshape(-1,1)
        for l in sorted(set(llist)):
            label = (l, q)
            tm_label_vals.append(label)
            components_count = 2*l+1
            properties_count = llist.count(l)
            blocks[label] = np.zeros((samples_count, components_count, properties_count))
            block_comp_label_vals[label] = np.arange(-l, l+1).reshape(-1,1)
            block_prop_label_vals[label] = np.arange(properties_count).reshape(-1,1)
            block_samp_label_vals[label] = block_samp_label_vals_q

    tm_labels = metatensor.Labels(vector_label_names.tm, np.array(tm_label_vals))

    block_comp_labels = {key: metatensor.Labels(vector_label_names.block_comp, block_comp_label_vals[key]) for key in blocks}
    block_prop_labels = {key: metatensor.Labels(vector_label_names.block_prop, block_prop_label_vals[key]) for key in blocks}
    block_samp_labels = {key: metatensor.Labels(vector_label_names.block_samp, block_samp_label_vals[key]) for key in blocks}

    # Fill in the blocks

    iq = dict.fromkeys(llists.keys(), 0)
    i = Cursor(action='slicer')
    for q in atom_charges:
        if llists[q]==sorted(llists[q]):
            for l in set(llists[q]):
                msize = 2*l+1
                nsize = blocks[(l,q)].shape[-1]
                cslice = c[i(nsize*msize)].reshape(nsize,msize).T
                if l==1:  # for l=1, the pyscf order is x,y,z (1,-1,0)
                    cslice = cslice[pyscf2gpr_l1_order]
                blocks[(l,q)][iq[q],:,:] = cslice
        else:
            il = dict.fromkeys(range(max(llists[q]) + 1), 0)
            for l in llists[q]:
                msize = 2*l+1
                cslice = c[i(msize)]
                if l==1:  # for l=1, the pyscf order is x,y,z (1,-1,0)
                    cslice = cslice[pyscf2gpr_l1_order]
                blocks[(l,q)][iq[q],:,il[l]] = cslice
                il[l] += 1
        iq[q] += 1

    # Build tensor blocks and tensor map

    tensor_blocks = [metatensor.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=[block_comp_labels[key]], properties=block_prop_labels[key]) for key in tm_label_vals]
    tensor = metatensor.TensorMap(keys=tm_labels, blocks=tensor_blocks)

    return tensor


def tensormap_to_vector(mol, tensor):
    """Transform a tensor map into a vector.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        tensor (metatensor.TensorMap): tensor to transform.

    Returns:
        numpy.ndarray: 1D array (vector) representation.

    Raises:
        RuntimeError: If tensor size does not match mol.nao.
    """
    nao = _get_tsize(tensor)
    if mol.nao != nao:
        raise RuntimeError(f'Tensor size mismatch ({nao} instead of {mol.nao})')

    c = np.zeros(mol.nao)
    atom_charges = mol.atom_charges()
    llists = _get_llist(mol)
    i = 0
    for iat, q in enumerate(atom_charges):
        llist = llists[q]
        il = dict.fromkeys(range(max(llist) + 1), 0)
        for l in llist:
            block = tensor.block(o3_lambda=l, center_type=q)
            id_samp = block.samples.position((iat,))
            id_prop = block.properties.position((il[l],))
            for m in get_mrange(l):
                id_comp = block.components[0].position((m,))
                c[i] = block.values[id_samp,id_comp,id_prop]
                i += 1
            il[l] += 1
    return c


def matrix_to_tensormap(mol, dm):
    """Transforms a matrix into a tensor map.

    Each element of the matrix corresponds to a pair of atomic orbitals.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        dm (numpy.ndarray): matrix to transform.

    Returns:
        metatensor.TensorMap: Tensor map representation of the matrix.
    """
    atom_charges = mol.atom_charges()
    elements, counts = np.unique(atom_charges, return_counts=True)
    counts = dict(zip(elements, counts, strict=True))
    element_indices = {q: np.where(atom_charges==q)[0] for q in elements}
    llists = _get_llist(mol)

    tm_label_vals = []
    block_prop_label_vals = {}
    block_samp_label_vals = {}
    block_comp_label_vals = {}
    blocks = {}

    # Create labels for TensorMap, lables for blocks, and empty blocks

    for q1 in elements:
        for q2 in elements:
            samples_count1 = counts[q1]
            samples_count2 = counts[q2]
            llist1 = llists[q1]
            llist2 = llists[q2]
            block_samp_label_vals_q1q2 = np.array([*itertools.product(element_indices[q1], element_indices[q2])])
            for l1 in sorted(set(llist1)):
                components_count1 = 2*l1+1
                properties_count1 = llist1.count(l1)
                for l2 in sorted(set(llist2)):
                    components_count2 = 2*l2+1
                    properties_count2 = llist2.count(l2)

                    label = (l1, l2, q1, q2)
                    tm_label_vals.append(label)
                    blocks[label] = np.zeros((samples_count1*samples_count2, components_count1, components_count2, properties_count1*properties_count2))
                    block_comp_label_vals[label] = (np.arange(-l1, l1+1).reshape(-1,1), np.arange(-l2, l2+1).reshape(-1,1))
                    block_prop_label_vals[label] = np.array([*itertools.product(np.arange(properties_count1), np.arange(properties_count2))])
                    block_samp_label_vals[label] = block_samp_label_vals_q1q2

    tm_labels = metatensor.Labels(matrix_label_names.tm, np.array(tm_label_vals))

    block_prop_labels = {key: metatensor.Labels(matrix_label_names.block_prop, block_prop_label_vals[key]) for key in blocks}
    block_samp_labels = {key: metatensor.Labels(matrix_label_names.block_samp, block_samp_label_vals[key]) for key in blocks}
    block_comp_labels = {key: [metatensor.Labels([name], vals) for name, vals in zip(matrix_label_names.block_comp, block_comp_label_vals[key], strict=True)] for key in blocks}

    # Build tensor blocks
    tensor_blocks = [metatensor.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=block_comp_labels[key], properties=block_prop_labels[key]) for key in tm_label_vals]

    # Fill in the blocks

    if all(llists[q]==sorted(llists[q]) for q in llists):
        iq1 = dict.fromkeys(elements, 0)
        i1 = Cursor(action='slicer')
        for iat1, q1 in enumerate(atom_charges):
            for l1 in set(llists[q1]):
                msize1 = 2*l1+1
                nsize1 = llists[q1].count(l1)
                iq2 = dict.fromkeys(elements, 0)
                i1.add(nsize1*msize1)
                i2 = Cursor(action='slicer')
                for iat2, q2 in enumerate(atom_charges):
                    for l2 in set(llists[q2]):
                        msize2 = 2*l2+1
                        nsize2 = llists[q2].count(l2)
                        dmslice = dm[i1(),i2(nsize2*msize2)].reshape(nsize1,msize1,nsize2,msize2)
                        dmslice = np.transpose(dmslice, axes=[1,3,0,2]).reshape(msize1,msize2,-1)
                        block = tensor_blocks[tm_label_vals.index((l1,l2,q1,q2))]
                        at_p = block.samples.position((iat1,iat2))
                        blocks[(l1,l2,q1,q2)][at_p,:,:,:] = dmslice
                    iq2[q2] += 1
            iq1[q1] += 1
    else:
        iq1 = dict.fromkeys(elements, 0)
        i1 = Cursor(action='slicer')
        for iat1, q1 in enumerate(atom_charges):
            il1 = dict.fromkeys(range(max(llists[q1]) + 1), 0)
            for l1 in llists[q1]:
                i1.add(2*l1+1)
                iq2 = dict.fromkeys(elements, 0)
                i2 = Cursor(action='slicer')
                for iat2, q2 in enumerate(atom_charges):
                    il2 = dict.fromkeys(range(max(llists[q2]) + 1), 0)
                    for l2 in llists[q2]:
                        dmslice = dm[i1(),i2(2*l2+1)]
                        block = tensor_blocks[tm_label_vals.index((l1, l2, q1, q2))]
                        at_p = block.samples.position((iat1, iat2))
                        n_p = block.properties.position((il1[l1], il2[l2]))
                        blocks[(l1,l2,q1,q2)][at_p,:,:,n_p] = dmslice
                        il2[l2] += 1
                    iq2[q2] += 1
                il1[l1] += 1
            iq1[q1] += 1

    # Fix the m order (for l=1, the pyscf order is x,y,z (1,-1,0))
    for key in blocks:
        l1,l2 = key[:2]
        if l1==1:
            blocks[key] = np.ascontiguousarray(blocks[key][:,pyscf2gpr_l1_order,:,:])
        if l2==1:
            blocks[key] = np.ascontiguousarray(blocks[key][:,:,pyscf2gpr_l1_order,:])

    # Build tensor map
    tensor_blocks = [metatensor.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=block_comp_labels[key], properties=block_prop_labels[key]) for key in tm_label_vals]
    tensor = metatensor.TensorMap(keys=tm_labels, blocks=tensor_blocks)

    return tensor


def tensormap_to_matrix(mol, tensor):
    """Transform a tensor map into a matrix.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        tensor (metatensor.TensorMap): tensor to transform.

    Returns:
        numpy.ndarray: 2D array (matrix) representation.

    Raises:
        RuntimeError: If tensor size does not match mol.nao * mol.nao.
    """
    nao2 = _get_tsize(tensor)
    if mol.nao*mol.nao != nao2:
        raise RuntimeError(f'Tensor size mismatch ({nao2} instead of {mol.nao*mol.nao})')

    dm = np.zeros((mol.nao, mol.nao))
    atom_charges = mol.atom_charges()
    llists = _get_llist(mol)
    i1 = 0
    for iat1, q1 in enumerate(atom_charges):
        llist1 = llists[q1]
        il1 = dict.fromkeys(range(max(llist1) + 1), 0)
        for l1 in llist1:
            for m1 in get_mrange(l1):
                i2 = 0
                for iat2, q2 in enumerate(atom_charges):
                    llist2 = llists[q2]
                    il2 = dict.fromkeys(range(max(llist2) + 1), 0)
                    for l2 in llist2:
                        block = tensor.block(o3_lambda1=l1, o3_lambda2=l2, center_type1=q1, center_type2=q2)
                        id_samp = block.samples.position((iat1, iat2))
                        id_prop = block.properties.position((il1[l1], il2[l2]))
                        for m2 in get_mrange(l2):
                            id_comp1 = block.components[0].position((m1,))
                            id_comp2 = block.components[1].position((m2,))
                            dm[i1, i2] = block.values[id_samp, id_comp1, id_comp2, id_prop]
                            i2 += 1
                        il2[l2] += 1
                i1 += 1
            il1[l1] += 1
    return dm


def array_to_tensormap(mol, v):
    """Transform an array into a tensor map.

    Wrapper for vector_to_tensormap and matrix_to_tensormap.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        v (numpy.ndarray): Array to transform. Can be a vector (1D) or matrix (2D).

    Returns:
        metatensor.TensorMap: Tensor map representation of the array.

    Raises:
        ValueError: If array dimension is not 1 or 2.
    """
    if v.ndim==1:
        return vector_to_tensormap(mol, v)
    elif v.ndim==2:
        return matrix_to_tensormap(mol, v)
    else:
        raise ValueError(f'Cannot convert to TensorMap an array with ndim={v.ndim}')


def tensormap_to_array(mol, tensor):
    """Transform a tensor map into an array.

    Wrapper for tensormap_to_vector and tensormap_to_matrix.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        tensor (metatensor.TensorMap): Tensor to transform.

    Returns:
        numpy.ndarray: Array representation (1D vector or 2D matrix).

    Raises:
        RuntimeError: If tensor key names don't match expected format.
    """
    if tensor.keys.names==vector_label_names.tm:
        return tensormap_to_vector(mol, tensor)
    elif tensor.keys.names==matrix_label_names.tm:
        return tensormap_to_matrix(mol, tensor)
    else:
        raise RuntimeError('Tensor key names mismatch. Cannot determine if it is a vector or a matrix')


def join(tensors):
    """Merge two or more tensors with the same label names avoiding information duplication.

    Args:
        tensors (list): List of metatensor.TensorMap objects.

    Returns:
        metatensor.TensorMap: Merged tensor containing information from all input tensors.

    Raises:
        RuntimeError: If tensors have different label names.
    """
    if not all(tensor.keys.names==tensors[0].keys.names for tensor in tensors):
        raise RuntimeError('Cannot merge tensors with different label names')
    tm_label_vals = set().union(*[set(_labels_to_array(tensor.keys)) for tensor in tensors])
    tm_label_vals = sorted(tuple(value) for value in tm_label_vals)
    tm_labels = metatensor.Labels(tensors[0].keys.names, np.array(tm_label_vals))

    blocks = {}
    block_comp_labels = {}
    block_prop_labels = {}
    block_samp_labels = {}
    block_samp_label_vals = {}

    for label in tm_labels:
        key = tuple(label.values)
        blocks[key] = []
        block_samp_label_vals[key] = []
        for imol,tensor in enumerate(tensors):
            if label not in tensor.keys:
                continue
            block = tensor.block(label)
            blocks[key].append(block.values)
            block_samp_label_vals[key].extend([(imol, *s) for s in block.samples])

            if key not in block_comp_labels:
                block_comp_labels[key] = block.components
                block_prop_labels[key] = block.properties

    for key in blocks:
        blocks[key] = np.concatenate(blocks[key])
        block_samp_label_vals[key] = np.array(block_samp_label_vals[key])
        block_samp_labels[key] = metatensor.Labels((_molid_name, *tensor.sample_names), block_samp_label_vals[key])

    tensor_blocks = [metatensor.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=block_comp_labels[key], properties=block_prop_labels[key]) for key in tm_label_vals]
    tensor = metatensor.TensorMap(keys=tm_labels, blocks=tensor_blocks)

    return tensor


def split(tensor):
    """Split a tensor based on the molecule information stored within the input TensorMap.

    Args:
        tensor (metatensor.TensorMap): Tensor containing several molecules.

    Returns:
        list or dict: Collection of metatensor.TensorMap objects, one per molecule.
        Returns list if molecule indices are continuous, dict otherwise.

    Raises:
        RuntimeError: If tensor does not contain multiple molecules.
    """
    if tensor.sample_names[0]!=_molid_name:
        raise RuntimeError('Tensor does not seem to contain several molecules')

    # Check if the molecule indices are continuous
    mollist = sorted(reduce(
        lambda a,b: a.union(b),
        [set(block.samples.column(_molid_name)) for block in tensor.blocks()],
    ))
    if mollist==list(range(len(mollist))):
        tensors = [None] * len(mollist)
    else:
        tensors = {}

    # Common labels
    block_comp_labels = {}
    block_prop_labels = {}
    for label,block in tensor.items():
        key = tuple(label.values)
        block = tensor.block(label)
        block_comp_labels[key] = block.components
        block_prop_labels[key] = block.properties

    # Tensors for each molecule
    for imol in mollist:
        blocks = {}
        block_samp_labels = {}

        for label in tensor.keys:
            key = tuple(label.values)
            block = tensor.block(label)

            samples = [(sample_i,lbl) for sample_i,lbl in enumerate(block.samples.values) if lbl[0]==imol]
            if len(samples)==0:
                continue
            sampleidx = [t[0] for t in samples]
            samplelbl = [t[1] for t in samples]
            #sampleidx = [block.samples.position(lbl) for lbl in samplelbl]

            blocks[key] = block.values[sampleidx]
            block_samp_labels[key] = metatensor.Labels(tensor.sample_names[1:], np.array(samplelbl)[:,1:])

        tm_label_vals = sorted(blocks.keys())
        tm_labels = metatensor.Labels(tensor.keys.names, np.array(tm_label_vals))
        tensor_blocks = [metatensor.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=block_comp_labels[key], properties=block_prop_labels[key]) for key in tm_label_vals]
        tensors[imol] = metatensor.TensorMap(keys=tm_labels, blocks=tensor_blocks)

    return tensors
