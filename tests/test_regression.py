#!/usr/bin/env python3

import os
import numpy as np
import qstack.regression.hyperparameters as hyperparameters
import qstack.regression.regression as regression
import qstack.regression.final_error as final_error
import qstack.regression.condition as condition
import qstack.regression.oos as oos
import qstack.regression.cross_validate_results as cv_results
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
    X = np.load(os.path.join(path, 'data/mols/X_lb.npy'))
    y = np.loadtxt(os.path.join(path, 'data/mols/dipole.dat'))

    lc = regression.regression(X, y, sigma=3.162278e+01, eta=1.000000e+00, debug=True)
    true_lc = [(1, 0.3858612698451199, 0.007817875402536187),
               (2, 0.3699378161120017, 0.10616950528187692),
               (4, 0.3136096022855527, 0.07485650465620536),
               (6, 0.24018169400891018, 0.08584295185009833),
               (8, 0.2708852104417901, 7.021666937153402e-17)]

    assert(np.allclose(lc, true_lc))


def test_regression_sparse():
    path = os.path.dirname(os.path.realpath(__file__))
    X = np.load(os.path.join(path, 'data/mols/X_lb.npy'))
    y = np.loadtxt(os.path.join(path, 'data/mols/dipole.dat'))

    lc = regression.regression(X, y, sigma=3.162278e+01, eta=1.000000e+00, debug=True, sparse=5)
    true_lc = [(1, 0.26949030187125617, 0.0037384513934352747),
               (2, 0.4522608435324484, 0.3148074318684532),
               (4, 0.4803773474666784, 0.19356070353924582),
               (6, 0.333707374435793, 0.13803898307368923),
               (8, 0.4501685644789055, 8.95090418262362e-17)]
    assert(np.allclose(lc, true_lc))


def test_regression_idx():
    path = os.path.dirname(os.path.realpath(__file__))
    X = np.load(os.path.join(path, 'data/mols/X_lb.npy'))
    y = np.loadtxt(os.path.join(path, 'data/mols/dipole.dat'))
    error0 = 0.6217848772505417
    # use sklearn
    error = regression.regression(X, y, sigma=3.162278e+01, eta=1.000000e+00, debug=True, train_size=[1.0], random_state=666)
    assert np.allclose(error[0][1], error0)
    idx_train, idx_test = [0, 9, 3, 4, 8, 1, 6, 2], [5,7]  # correspond to this random state
    # pass test idx
    error = regression.regression(X, y, sigma=3.162278e+01, eta=1.000000e+00, debug=True, train_size=[1.0], idx_test=idx_test)
    assert np.allclose(error[0][1], error0)
    # pass train idx
    error = regression.regression(X, y, sigma=3.162278e+01, eta=1.000000e+00, debug=True, train_size=[1.0], idx_train=idx_train)
    assert np.allclose(error[0][1], error0)
    # pass both idx
    error = regression.regression(X, y, sigma=3.162278e+01, eta=1.000000e+00, debug=True, train_size=[1.0], idx_test=idx_test, idx_train=idx_train)
    assert np.allclose(error[0][1], error0)
    # pass negative idx
    error = regression.regression(X, y, sigma=3.162278e+01, eta=1.000000e+00, debug=True, train_size=[1.0], idx_test=np.array(idx_test)-len(y))
    assert np.allclose(error[0][1], error0)


def test_final_error():
    path = os.path.dirname(os.path.realpath(__file__))
    X = np.load(os.path.join(path, 'data/mols/X_lb.npy'))
    y = np.loadtxt(os.path.join(path, 'data/mols/dipole.dat'))
    error0 = regression.regression(X, y, sigma=3.162278e+01, eta=1.000000e+00, debug=True, train_size=[1.0], random_state=666)[0][1]
    mol_errors = final_error.final_error(X, y, sigma=3.162278e+01, eta=1.000000e+00, random_state=666)
    assert np.allclose(mol_errors.mean(), error0)


def test_cond():
    path = os.path.dirname(os.path.realpath(__file__))
    X = np.load(os.path.join(path, 'data/mols/X_lb.npy'))
    c = condition.condition(X, sigma=3.162278e+01, eta=1.000000e+00, random_state=0)
    c0 = 7.071059858021516
    assert np.allclose(c, c0)


def test_oos():
    path = os.path.dirname(os.path.realpath(__file__))
    X = np.load(os.path.join(path, 'data/mols/X_lb.npy'))
    y = np.loadtxt(os.path.join(path, 'data/mols/dipole.dat'))
    _, pred1, weights = final_error.final_error(X, y, sigma=3.162278e+01, eta=1e-10, random_state=666, return_pred=True, return_alpha=True)
    idx_train, idx_test = [0, 9, 3, 4, 8, 1, 6, 2], [5,7]  # correspond to this random state
    pred2 = oos.oos(X, X[idx_test], weights, sigma=3.162278e+01, random_state=666)
    assert np.allclose(pred1, pred2)
    pred3 = oos.oos(X, X[idx_train], weights, sigma=3.162278e+01, random_state=666)
    assert np.allclose(pred3, y[idx_train])

def test_cross_validate_results():
    path = os.path.dirname(os.path.realpath(__file__))
    X = np.load(os.path.join(path, 'data/mols/X_lb.npy'))
    y = np.loadtxt(os.path.join(path, 'data/mols/dipole.dat'))
    lc = cv_results.cv_results(X, y)
    true_lc = [(1, 0.96457399, 0.70560469),
               (2, 0.78990697, 0.58179988),
               (4, 0.7336549 , 0.59839317),
               (6, 0.7288867 , 0.50714861),
               (8, 0.72604955, 0.48307486)]
    assert(np.allclose(lc, true_lc))



if __name__ == '__main__':
    test_generate_reps()
    test_hyperparameters()
    test_regression()
    test_regression_sparse()
    test_regression_idx()
    test_final_error()
    test_cond()
    test_oos()
