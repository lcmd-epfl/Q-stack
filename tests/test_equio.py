#!/usr/bin/env python3

import os
import numpy as np
from qstack import compound, fields
from qstack.fields.decomposition import decompose, correct_N, correct_N_atomic
from qstack.fields.density2file import coeffs_to_cube, coeffs_to_molden
import pyscf

import equistore

def test_equio():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp', charge=0, spin=0)
    dm   = fields.dm.get_converged_dm(mol, xc="pbe")
    auxmol, c = decompose(mol, dm, 'sto3g')
    auxmol, c = decompose(mol, dm, 'cc-pvdz jkfit')
    print()
    print()

#    print(c)

    atom_charges = list(auxmol.atom_charges())
    elements = sorted(set(atom_charges))


    ###########################
    # Create:
    # labels for TensorMap,
    # lables for blocks
    # empty blocks
    ###########################

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

    for q in elements:
        qname = pyscf.data.elements.ELEMENTS[q]
        llist = [b[0] for b in auxmol._basis[qname]]
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

    ###########################
    # Fill in the blocks
    ###########################

    iq = {q:0 for q in elements}
    i = 0
    for iat, q in enumerate(atom_charges):
        il = {l:0 for l in range(max(llists[q])+1)}
        for l in llists[q]:
            msize = 2*l+1
            if l!=1:
                cslice = c[i:i+msize]
            else:
                # for l=1, the pyscf order is x,y,z (1,-1,0)
                cslice = c[i+1], c[i+2], c[i]
            blocks [(l,q)] [ iq[q] , : , il[l] ] = cslice
            i     += msize
            il[l] += 1
        iq[q] += 1

    ####################################
    # Build tensor blocks and tensor map
    ####################################

    tensor_blocks = [equistore.TensorBlock(values=blocks[key], samples=block_samp_labels[key], components=[block_comp_labels[key]], properties=block_prop_labels[key]) for key in tm_label_vals]
    tensor = equistore.TensorMap(keys=tm_labels, blocks=tensor_blocks)

    #########################

    print(tensor)


#
#N = fields.decomposition.number_of_electrons_deco(auxmol, c)
#
#print("Number of electrons after decomposition: ", N)
#
#coeffs_to_cube(auxmol, c, 'H2O.cube')
#print('density saved to H2O.cube')
#
#coeffs_to_molden(auxmol, c, 'H2O.molden')
#print('density saved to H2O.molden')
#
#c = correct_N(auxmol, c)
#N = fields.decomposition.number_of_electrons_deco(auxmol, c)
#print(N)
#
#
#c = correct_N_atomic(auxmol, np.array([8,1,1]), c)

if __name__ == '__main__':
    test_equio()
