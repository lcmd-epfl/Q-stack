import numpy as np
from pyscf import data
import equistore

def vector_to_tensormap(mol, c):

    atom_charges = list(mol.atom_charges())
    elements = sorted(set(atom_charges))

    tm_label_names  = ['spherical_harmonics_l', 'element']
    tm_label_vals = []

    block_prop_label_names  = ['radial_channel']
    block_prop_label_vals = {}

    block_samp_label_names  = ['atom_id']
    block_samp_label_vals = {}

    block_comp_label_names  = ['spherical_harmonics_m']
    block_comp_label_vals = {}

    blocks = {}
    llists = {}

    # Create labels for TensorMap, lables for blocks, and empty blocks

    for q in elements:
        qname = data.elements.ELEMENTS[q]
        llist = [b[0] for b in mol._basis[qname]]
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

    tm_labels = equistore.Labels(tm_label_names, np.array(tm_label_vals))

    block_comp_labels = {key: equistore.Labels(block_comp_label_names, block_comp_label_vals[key]) for key in blocks}
    block_prop_labels = {key: equistore.Labels(block_prop_label_names, block_prop_label_vals[key]) for key in blocks}
    block_samp_labels = {key: equistore.Labels(block_samp_label_names, block_samp_label_vals[key]) for key in blocks}

    # Fill in the blocks

    iq = {q:0 for q in elements}
    i = 0
    for iat, q in enumerate(atom_charges):
        if llists[q]==sorted(llists[q]):
            for l in set(llists[q]):
                msize = 2*l+1
                nsize = blocks[(l,q)].shape[-1]
                cslice =  c[i:i+nsize*msize].reshape(nsize,msize).T
                if l==1: # for l=1, the pyscf order is x,y,z (1,-1,0)
                    cslice = cslice[[1,2,0]]
                blocks [(l,q)] [ iq[q] , : , : ] = cslice
                i += msize*nsize
        else:
            il = {l:0 for l in range(max(llists[q])+1)}
            for l in llists[q]:
                msize = 2*l+1
                cslice = c[i:i+msize]
                if l==1: # for l=1, the pyscf order is x,y,z (1,-1,0)
                    cslice = cslice[[1,2,0]]
                blocks [(l,q)] [ iq[q] , : , il[l] ] = cslice
                i     += msize
                il[l] += 1
        iq[q] += 1

    # Build tensor blocks and tensor map

    tensor_blocks = [equistore.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=[block_comp_labels[key]], properties=block_prop_labels[key]) for key in tm_label_vals]
    tensor = equistore.TensorMap(keys=tm_labels, blocks=tensor_blocks)

    return tensor


def tensormap_to_vector(mol, tensor):
    i=0
    c = np.zeros(mol.nao)
    atom_charges = mol.atom_charges()
    for iat, q in enumerate(atom_charges):
        qname = data.elements.ELEMENTS[q]
        llist = [b[0] for b in mol._basis[qname]]
        il = {l:0 for l in range(max(llist)+1)}
        for l in llist:
            block = tensor.block(spherical_harmonics_l=l, element=q)
            id_samp = block.samples.position((iat,))
            id_prop = block.properties.position((il[l],))
            if l==1: # for l=1, the pyscf order is x,y,z (1,-1,0)
                mrange = (1,-1,0)
            else:
                mrange = range(-l,l+1)
            for m in mrange:
                id_comp = block.components[0].position((m,))
                c[i] = block.values[id_samp,id_comp,id_prop]
                i += 1
            il[l] += 1
    return c
