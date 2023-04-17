import numpy as np
from types import SimpleNamespace
from pyscf import data
import equistore
import numbers

vector_label_names = SimpleNamespace(
    tm = ['spherical_harmonics_l', 'element'],
    block_prop = ['radial_channel'],
    block_samp = ['atom_id'],
    block_comp = ['spherical_harmonics_m']
    )

matrix_label_names = SimpleNamespace(
    tm = ['spherical_harmonics_l1', 'spherical_harmonics_l2', 'element1', 'element2'],
    block_prop = ['radial_channel1', 'radial_channel2'],
    block_samp = ['atom_id1', 'atom_id2'],
    block_comp = ['spherical_harmonics_m1', 'spherical_harmonics_m2']
    )

_molid_name = 'mol_id'

_pyscf2gpr_l1_order = [1,2,0]


def _get_mrange(l):
    # for l=1, the pyscf order is x,y,z (1,-1,0)
    if l==1:
        return (1,-1,0)
    else:
        return range(-l,l+1)


def _get_llist(q, mol):
    # TODO other basis formats?
#        for bas_id in mol.atom_shell_ids(iat):
#            l  = mol.bas_angular(bas_id)
#            nc = mol.bas_nctr(bas_id)
#            for n in range(nc):
    if isinstance(q, numbers.Integral):
        q = data.elements.ELEMENTS[q]
    llist = []
    for l, *prim in mol._basis[q]:
        llist.extend([l]*(len(prim[0])-1))
    return llist


def _get_tsize(tensor):
    return sum([np.prod(tensor.block(key).values.shape) for key in tensor.keys])


def vector_to_tensormap(mol, c):

    atom_charges = list(mol.atom_charges())
    elements = sorted(set(atom_charges))

    tm_label_vals = []
    block_prop_label_vals = {}
    block_samp_label_vals = {}
    block_comp_label_vals = {}

    blocks = {}
    llists = {}

    # Create labels for TensorMap, lables for blocks, and empty blocks

    for q in elements:
        llist = _get_llist(q, mol)
        llists[q] = llist
        for l in sorted(set(llist)):
            label = (l, q)
            tm_label_vals.append(label)
            samples_count    = atom_charges.count(q)
            components_count = 2*l+1
            properties_count = llist.count(l)
            blocks[label] = np.zeros((samples_count, components_count, properties_count))
            block_comp_label_vals[label] = np.arange(-l, l+1).reshape(-1,1)
            block_prop_label_vals[label] = np.arange(properties_count).reshape(-1,1)
            block_samp_label_vals[label] = np.where(atom_charges==q)[0].reshape(-1,1)

    tm_labels = equistore.Labels(vector_label_names.tm, np.array(tm_label_vals))

    block_comp_labels = {key: equistore.Labels(vector_label_names.block_comp, block_comp_label_vals[key]) for key in blocks}
    block_prop_labels = {key: equistore.Labels(vector_label_names.block_prop, block_prop_label_vals[key]) for key in blocks}
    block_samp_labels = {key: equistore.Labels(vector_label_names.block_samp, block_samp_label_vals[key]) for key in blocks}

    # Fill in the blocks

    iq = {q:0 for q in elements}
    i = 0
    for iat, q in enumerate(atom_charges):
        if llists[q]==sorted(llists[q]):
            for l in set(llists[q]):
                msize = 2*l+1
                nsize = blocks[(l,q)].shape[-1]
                cslice = c[i:i+nsize*msize].reshape(nsize,msize).T
                if l==1:  # for l=1, the pyscf order is x,y,z (1,-1,0)
                    cslice = cslice[_pyscf2gpr_l1_order]
                blocks[(l,q)][iq[q],:,:] = cslice
                i += msize*nsize
        else:
            il = {l:0 for l in range(max(llists[q])+1)}
            for l in llists[q]:
                msize = 2*l+1
                cslice = c[i:i+msize]
                if l==1:  # for l=1, the pyscf order is x,y,z (1,-1,0)
                    cslice = cslice[_pyscf2gpr_l1_order]
                blocks[(l,q)][iq[q],:,il[l]] = cslice
                i     += msize
                il[l] += 1
        iq[q] += 1

    # Build tensor blocks and tensor map

    tensor_blocks = [equistore.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=[block_comp_labels[key]], properties=block_prop_labels[key]) for key in tm_label_vals]
    tensor = equistore.TensorMap(keys=tm_labels, blocks=tensor_blocks)

    return tensor


def tensormap_to_vector(mol, tensor):

    nao = _get_tsize(tensor)
    if mol.nao != nao:
        raise Exception(f'Tensor size mismatch ({nao} instead of {mol.nao})')

    c = np.zeros(mol.nao)
    atom_charges = mol.atom_charges()
    i = 0
    for iat, q in enumerate(atom_charges):
        llist = _get_llist(q, mol)
        il = {l: 0 for l in range(max(llist)+1)}
        for l in llist:
            block = tensor.block(spherical_harmonics_l=l, element=q)
            id_samp = block.samples.position((iat,))
            id_prop = block.properties.position((il[l],))
            for m in _get_mrange(l):
                id_comp = block.components[0].position((m,))
                c[i] = block.values[id_samp,id_comp,id_prop]
                i += 1
            il[l] += 1
    return c


def matrix_to_tensormap(mol, dm):

    def pairs(list1, list2):
        return np.array([(i,j) for i in list1 for j in list2])

    atom_charges = list(mol.atom_charges())
    elements = sorted(set(atom_charges))

    tm_label_vals = []
    block_prop_label_vals = {}
    block_samp_label_vals = {}
    block_comp_label_vals = {}

    blocks = {}
    llists = {q: _get_llist(q, mol) for q in elements}

    # Create labels for TensorMap, lables for blocks, and empty blocks

    for q1 in elements:
        for q2 in elements:
            llist1 = llists[q1]
            llist2 = llists[q2]
            for l1 in sorted(set(llist1)):
                for l2 in sorted(set(llist2)):
                    label = (l1, l2, q1, q2)
                    tm_label_vals.append(label)

                    samples_count1    = atom_charges.count(q1)
                    components_count1 = 2*l1+1
                    properties_count1 = llist1.count(l1)

                    samples_count2    = atom_charges.count(q2)
                    components_count2 = 2*l2+1
                    properties_count2 = llist2.count(l2)

                    blocks[label] = np.zeros((samples_count1*samples_count2, components_count1, components_count2, properties_count1*properties_count2))
                    block_comp_label_vals[label] = (np.arange(-l1, l1+1).reshape(-1,1), np.arange(-l2, l2+1).reshape(-1,1))
                    block_prop_label_vals[label] = pairs(np.arange(properties_count1), np.arange(properties_count2))
                    block_samp_label_vals[label] = pairs(np.where(atom_charges==q1)[0],np.where(atom_charges==q2)[0])

    tm_labels = equistore.Labels(matrix_label_names.tm, np.array(tm_label_vals))

    block_prop_labels = {key: equistore.Labels(matrix_label_names.block_prop, block_prop_label_vals[key]) for key in blocks}
    block_samp_labels = {key: equistore.Labels(matrix_label_names.block_samp, block_samp_label_vals[key]) for key in blocks}
    block_comp_labels = {key: [equistore.Labels([name], vals) for name, vals in zip(matrix_label_names.block_comp, block_comp_label_vals[key])] for key in blocks}

    # Build tensor blocks
    tensor_blocks = [equistore.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=block_comp_labels[key], properties=block_prop_labels[key]) for key in tm_label_vals]

    # Fill in the blocks

    if all([llists[q]==sorted(llists[q]) for q in llists]):
        iq1 = {q1: 0 for q1 in elements}
        i1 = 0
        for iat1, q1 in enumerate(atom_charges):
            for l1 in set(llists[q1]):
                msize1 = 2*l1+1
                nsize1 = llists[q1].count(l1)
                iq2 = {q2: 0 for q2 in elements}
                i2 = 0
                for iat2, q2 in enumerate(atom_charges):
                    for l2 in set(llists[q2]):
                        msize2 = 2*l2+1
                        nsize2 = llists[q2].count(l2)
                        dmslice = dm[i1:i1+nsize1*msize1,i2:i2+nsize2*msize2].reshape(nsize1,msize1,nsize2,msize2)
                        dmslice = np.transpose(dmslice, axes=[1,3,0,2]).reshape(msize1,msize2,-1)
                        block = tensor_blocks[tm_label_vals.index((l1,l2,q1,q2))]
                        at_p = block.samples.position((iat1,iat2))
                        blocks[(l1,l2,q1,q2)][at_p,:,:,:] = dmslice
                        i2 += msize2*nsize2
                    iq2[q2] += 1
                i1 += msize1*nsize1
            iq1[q1] += 1
    else:
        iq1 = {q1: 0 for q1 in elements}
        i1 = 0
        for iat1, q1 in enumerate(atom_charges):
            il1 = {l1: 0 for l1 in range(max(llists[q1])+1)}
            for l1 in llists[q1]:
                msize1 = 2*l1+1
                iq2 = {q2: 0 for q2 in elements}
                i2 = 0
                for iat2, q2 in enumerate(atom_charges):
                    il2 = {l2: 0 for l2 in range(max(llists[q2])+1)}
                    for l2 in llists[q2]:
                        msize2 = 2*l2+1
                        dmslice = dm[i1:i1+msize1,i2:i2+msize2]
                        block = tensor_blocks[tm_label_vals.index((l1, l2, q1, q2))]
                        at_p = block.samples.position((iat1, iat2))
                        n_p = block.properties.position((il1[l1], il2[l2]))
                        blocks[(l1,l2,q1,q2)][at_p,:,:,n_p] = dmslice
                        i2 += msize2
                        il2[l2] += 1
                    iq2[q2] += 1
                i1 += msize1
                il1[l1] += 1
            iq1[q1] += 1

    # Fix the m order (for l=1, the pyscf order is x,y,z (1,-1,0))
    for key in blocks:
        l1,l2 = key[:2]
        if l1==1:
            blocks[key] = np.ascontiguousarray(blocks[key][:,_pyscf2gpr_l1_order,:,:])
        if l2==1:
            blocks[key] = np.ascontiguousarray(blocks[key][:,:,_pyscf2gpr_l1_order,:])

    # Build tensor map
    tensor_blocks = [equistore.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=block_comp_labels[key], properties=block_prop_labels[key]) for key in tm_label_vals]
    tensor = equistore.TensorMap(keys=tm_labels, blocks=tensor_blocks)

    return tensor


def tensormap_to_matrix(mol, tensor):

    nao2 = _get_tsize(tensor)
    if mol.nao*mol.nao != nao2:
        raise Exception(f'Tensor size mismatch ({nao2} instead of {mol.nao*mol.nao})')

    dm = np.zeros((mol.nao, mol.nao))
    atom_charges = mol.atom_charges()
    i1 = 0
    for iat1, q1 in enumerate(atom_charges):
        llist1 = _get_llist(q1, mol)
        il1 = {l1: 0 for l1 in range(max(llist1)+1)}
        for l1 in llist1:
            for m1 in _get_mrange(l1):

                i2 = 0
                for iat2, q2 in enumerate(atom_charges):
                    llist2 = _get_llist(q2, mol)
                    il2 = {l2: 0 for l2 in range(max(llist2)+1)}
                    for l2 in llist2:

                        block = tensor.block(spherical_harmonics_l1=l1, spherical_harmonics_l2=l2, element1=q1, element2=q2)
                        id_samp = block.samples.position((iat1, iat2))
                        id_prop = block.properties.position((il1[l1], il2[l2]))

                        for m2 in _get_mrange(l2):
                            id_comp1 = block.components[0].position((m1,))
                            id_comp2 = block.components[1].position((m2,))
                            dm[i1, i2] = block.values[id_samp, id_comp1, id_comp2, id_prop]
                            i2 += 1
                        il2[l2] += 1
                i1 += 1
            il1[l1] += 1

    return dm

def array_to_tensormap(mol, v):
    if v.ndim==1:
        return vector_to_tensormap(mol, v)
    elif v.ndim==2:
        return matrix_to_tensormap(mol, v)
    else:
        raise Exception(f'Cannot convert to TensorMap an array with ndim={v.ndim}')


def tensormap_to_array(mol, tensor):
    if tensor.keys.names==tuple(vector_label_names.tm):
        return tensormap_to_vector(mol, tensor)
    elif tensor.keys.names==tuple(matrix_label_names.tm):
        return tensormap_to_matrix(mol, tensor)
    else:
        raise Exception(f'Tensor key names mismatch. Cannot determine if it is a vector or a matrix')


def join(tensors):

    if not all(tensor.keys.names==tensors[0].keys.names for tensor in tensors):
        raise Exception(f'Cannot merge tensors with different label names')
    tm_label_vals = sorted(list(set().union(*[set(tensor.keys.tolist()) for tensor in tensors])))
    tm_labels = equistore.Labels(tensors[0].keys.names, np.array(tm_label_vals))

    blocks = {}
    block_comp_labels = {}
    block_prop_labels = {}
    block_samp_labels = {}
    block_samp_label_vals = {}

    for label in tm_labels:
        key = tuple(label.tolist())
        blocks[key] = []
        block_samp_label_vals[key] = []
        for imol,tensor in enumerate(tensors):
            if not label in tensor.keys:
                continue
            block = tensor.block(label)
            blocks[key].append(block.values)
            block_samp_label_vals[key].extend([(imol, *s) for s in block.samples])

            if not key in block_comp_labels:
                block_comp_labels[key] = block.components
                block_prop_labels[key] = block.properties

    for key in blocks:
        blocks[key] = np.concatenate(blocks[key])
        block_samp_label_vals[key] = np.array(block_samp_label_vals[key])
        block_samp_labels[key] = equistore.Labels((_molid_name, *tensor.sample_names), block_samp_label_vals[key])

    tensor_blocks = [equistore.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=block_comp_labels[key], properties=block_prop_labels[key]) for key in tm_label_vals]
    tensor = equistore.TensorMap(keys=tm_labels, blocks=tensor_blocks)

    return tensor


def split(tensor):

    if tensor.sample_names[0]!=_molid_name:
        raise Exception(f'Tensor does not seem to contain several molecules')

    # Check if the molecule indices are continuous
    mollist = sorted(set(np.hstack([np.array(tensor.block(keys).samples.tolist())[:,0] for keys in tensor.keys])))
    if mollist==list(range(len(mollist))):
        tensors = [None] * len(mollist)
    else:
        tensors = {}

    # Common labels
    block_comp_labels = {}
    block_prop_labels = {}
    for label in tensor.keys:
        key = label.tolist()
        block = tensor.block(label)
        block_comp_labels[key] = block.components
        block_prop_labels[key] = block.properties

    # Tensors for each molecule
    for imol in mollist:
        blocks = {}
        block_samp_labels = {}

        for label in tensor.keys:
            key = label.tolist()
            block = tensor.block(label)

            samplelbl = [lbl for lbl in block.samples.tolist() if lbl[0]==imol]
            if len(samplelbl)==0:
                continue
            sampleidx = [block.samples.position(lbl) for lbl in samplelbl]

            blocks[key] = block.values[sampleidx]
            block_samp_labels[key] = equistore.Labels(tensor.sample_names[1:], np.array(samplelbl)[:,1:])

        tm_label_vals = sorted(list(blocks.keys()))
        tm_labels = equistore.Labels(tensor.keys.names, np.array(tm_label_vals))
        tensor_blocks = [equistore.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=block_comp_labels[key], properties=block_prop_labels[key]) for key in tm_label_vals]
        tensors[imol] = equistore.TensorMap(keys=tm_labels, blocks=tensor_blocks)

    return tensors
