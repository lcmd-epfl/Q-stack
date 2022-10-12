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
    blocks = {}

    atom_charges = list(auxmol.atom_charges())
    elements = sorted(set(atom_charges))
    lmax = {}
    llists = {}
    for q in elements:
        qname = pyscf.data.elements.ELEMENTS[q]
        llist = [b[0] for b in auxmol._basis[qname]]
        lmax[q] = max(llist)
        llists[q] = llist
        for l in sorted(set(llist)):
            label = (l, q)
            tm_label_values.append(label)
            samples_count    = atom_charges.count(q)
            components_count = 2*l+1
            properties_count = llist.count(l)
            blocks[label] = np.zeros((samples_count, components_count, properties_count))
    tm_labels = equistore.Labels(names=tm_label_names, values = np.array(tm_label_values))


    # fill the blocks
    iq = {q:0 for q in elements}
    i=0
    for iat, q in enumerate(atom_charges):
        il = {l:0 for l in range(max(llists[q])+1)}
        for l in llists[q]:
            msize = 2*l+1
            blocks [(l,q)] [ iq[q] , : , il[l] ] = c[i:i+msize]
            i     += msize
            il[l] += 1
        iq[q] += 1





    print()
    print()
    for i in blocks:
        print()
        print(i, blocks[i].shape)
        print(blocks[i])




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