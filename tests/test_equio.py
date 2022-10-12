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



    # labels to create TensorMap
    tm_label_names  = ['spherical_harmonics_l', 'element']
    tm_label_values = []
    empty_blocks = {}
    elements = list(auxmol.atom_charges())
    for q in sorted(set(elements)):
        qname = pyscf.data.elements.ELEMENTS[q]
        llist = [b[0] for b in auxmol._basis[qname]]
        for l in sorted(set(llist)):
            label = (l, q)
            tm_label_values.append(label)
            samples_count    = elements.count(q)
            components_count = 2*l+1
            properties_count = llist.count(l)
            empty_blocks[label] = np.zeros((samples_count, components_count, properties_count))
    tm_labels = equistore.Labels(names=tm_label_names, values = np.array(tm_label_values))


    # blocks
    for i in empty_blocks:
        print(i, empty_blocks[i].shape)


    block_lists = {}
    for i in empty_blocks:
        block_lists[i] = [[] for _ in range(empty_blocks[i].shape[0])]
    iq = {}
    for q in sorted(set(elements)):
        iq[q] = 0
    i=0
    for iat, q in enumerate(auxmol.atom_charges()):
        qname = pyscf.data.elements.ELEMENTS[q]
        for gto in auxmol._basis[qname]:
            l = gto[0]
            msize = 2*l+1
            block_lists[(l,q)][iq[q]].append(c[i:i+msize])
            i  += msize
        iq[q] += 1
    for i in block_lists:
        block_lists[i] = np.transpose(np.array(block_lists[i]), axes=(0,2,1))



    iq = {}
    for q in sorted(set(elements)):
        iq[q] = 0
    i=0
    for iat, q in enumerate(auxmol.atom_charges()):
        qname = pyscf.data.elements.ELEMENTS[q]
        il = {}
        for l in range(max([b[0] for b in auxmol._basis[qname]]) + 1):
            il[l] = 0
        for gto in auxmol._basis[qname]:
            l = gto[0]
            msize = 2*l+1
            empty_blocks [(l,q)] [iq[q], : , il[l]] = c[i:i+msize]
            i  += msize
            il[l] += 1
        iq[q] += 1





    print()
    print()
    for i in block_lists:
        print()
        print(np.linalg.norm(block_lists[i]-empty_blocks[i]))
        #print(i, block_lists[i].shape)




    #block = equistore.TensorBlock(values=, samples=, components=, properties=)



#    tensor = equistore.TensorMap(keys=labels, blocks=[])


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
