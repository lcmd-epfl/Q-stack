#!/usr/bin/env python3

import sys,os
from os.path import join, isfile, isdir
import numpy as np
from DMbRep import check_file


# Q-stack te permet de generer l'objet de base pour la molecule et utilise Pyscf pour obtenir l'hamiltonian et les veteurs propres.
from  qstack import compound, spahm
# Dans le dossier './modules/' tu retrouves toutes les fonctions propre au projet pour generer les representations
from modules import utils, dmb_rep_atom as dmba


def main() :

# En premier on recupere le guess Hamiltonian a partir d'ancien script qu'on a integre a Q-stach
# le meilleur des Hamiltonian est le Laikov-Brilling (LB) (https://github.com/lcmd-epfl/Q-stack/blob/master/qstack/spahm/LB2020guess.py)
# (https://github.com/lcmd-epfl/Q-stack/blob/master/qstack/spahm/guesses.py)

    guess = spahm.guesses.get_guess('LB')

# Ensuite on recupere le fonctions pour generer le bon model (on en avait imagine plusieurs) le seul qui marche vraiment 
# et qu'on utilise c'est le 'Lowdin-long-x'.

# La fonction get_a_DF() (modules/make_atomic_DF.py) produit les coefficients du density-fitting
# La fonction coefficients_symmetrize_long() (modules/dmb_rep_atom.py) applique l'invariance par rotation
# et garanti la taille du vecteur final vis-a-vis des especes atomiques (arguments)
# le tout est wrap dans get_model() (modules/dmb_rep_atom.py)

    model         = dmba.get_model('lowdin-long-x')
    df_wrapper, sym_wrapper = model

# Apres on recupere les info conercernant les basis qu'on utilises (index, slice d'atome, tailles des bases par atomes etc..)
    atom_types = ['H', 'O']         # le type d'atom present dans la database, toute molecules confondues, ici pour le test (H2O) je
                                    # mets que 'H' et 'O', les atomes qui sont pas presents dans la molecules sont padder de zeros

    aux_basis_set='ccpvdzjkfit'     # le basis-set qu'on utilise pour le fitting de la density-matrix
    ao, ao_len, idx, M = dmba.get_basis_info(atom_types, aux_basis_set)

### A partir d'ici on compute ###

# D'abord t'importes le fichier .xyz et on le check et on genere le Compound class de chey PySCF
# On utilise un ancien script qu'on a integre a Q-Stack (https://github.com/lcmd-epfl/Q-stack/blob/master/qstack/compound.py)

    mol_file = './test_spahm/H2O.xyz'
    check_file(mol_file)
    basis_set='minao'               # le basis set qu'on utilise pour generer le guess Hamiltonian (minimum c'est le best) 
    spin=None                       # si spin est pas 'None' on run en open-shell et ensuite
                                    # on concatene 'alpha' et 'beta' spin dans la rep finale


    mol = compound.xyz_to_mol(mol_file, basis_set, charge=0, spin=spin)

# On commence par calculer la density-matrix (la matrice de tout les produits scalaires des vecteurs propres) 


    DM = spahm.compute_spahm.get_guess_dm(mol, guess, openshell=spin)
# Apres on calcule les coefficients du density-fitting (a partir du wrapper)

    c_df    = df_wrapper(mol, DM, basis_set, aux_basis_set)

# Et on applique l'invariance par rotation ici (en suivant la derivation que je t'ai montre) (a partir du wrapper)
    vectors = sym_wrapper(c_df, mol, idx, ao, ao_len, M, atom_types)
    vectors = np.array(vectors, dtype=object)

# Dans cet array final tu retrouve un tuple pour chacun des atomes dans la molecule avec le type de l'atom en premiere
# position et la representation pour cet atom en second position
    print(vectors.shape)


# J'ai aussi creer un test pour l'eau au cas ou tu veux changer quelque chose dans les scripts et verifier que ca marche toujours

    from test_spahm import test_atomic_spahm as test
    test.test_water(vectors)

if __name__ == '__main__' : main()


