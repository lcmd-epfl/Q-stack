#!/usr/bin/env python3

import os
import numpy as np
import qstack.regression.hyperparameters as hyperparameters
import qstack.regression.regression as regression
import qstack.spahm.compute_spahm as espahm
import qstack.compound as compound

def test_generate_reps():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/mols/')
    xyzlist = [os.path.join(path,s) for s in sorted(os.listdir(path)) if ".xyz" in s]
    mols = [compound.xyz_to_mol(f, basis='minao', charge=0, spin=0) for f in xyzlist]
    xmols = [espahm.get_spahm_representation(mol, 'lb')[0] for mol in mols]
    maxlen = max([len(x) for x in xmols])
    X = np.array([np.pad(x, pad_width=(0,maxlen-len(x)), constant_values=0) for x in xmols])
    xfile = os.path.join(path, 'X_lb.npy')
    Xtrue = np.load(xfile)
    #print(xyzlist)
    assert(np.allclose(X,Xtrue))

def test_hyperparameters():
    path = os.path.dirname(os.path.realpath(__file__))
    xfile = os.path.join(path, 'data/mols/X_lb.npy')
    X = np.load(xfile)
    yfile = os.path.join(path, 'data/mols/dipole.dat')
    y = np.loadtxt(yfile)
    
    hyper = hyperparameters.hyperparameters(X, y)[-4:]
    true_hyper = [  [5.18303885e-01,3.00507798e-01,1.00000000e-05,3.16227766e+01],
                    [5.18262897e-01,3.00473853e-01,3.16227766e-08,3.16227766e+01],
                    [5.18262767e-01,3.00473746e-01,1.00000000e-10,3.16227766e+01],
                    [5.10592542e-01,3.38247735e-01,1.00000000e+00,3.16227766e+01]]

    assert(np.allclose(hyper, true_hyper))

def test_regression():
    path = os.path.dirname(os.path.realpath(__file__))
    xfile = os.path.join(path, 'data/mols/X_lb.npy')
    X = np.load(xfile)
    yfile = os.path.join(path, 'data/mols/dipole.dat')
    y = np.loadtxt(yfile)
    
    lc = regression.regression(X, y, sigma=3.162278e+01, eta=1.000000e+00, debug=True)
    true_lc = [(1, 0.3858612698451199, 0.007817875402536187), (2, 0.3699378161120017, 0.10616950528187692),
               (4, 0.3136096022855527, 0.07485650465620536), (6, 0.24018169400891018, 0.08584295185009833),
               (8, 0.2708852104417901, 7.021666937153402e-17)]

    assert(np.allclose(lc, true_lc))

def main():
    test_generate_reps()
    test_hyperparameters()
    test_regression()

if __name__ == '__main__': main()
