#!/usr/bin/env python3

import os
import tempfile
import numpy as np
from qstack.spahm.rho import compute_rho_spahm as rho


path = os.path.dirname(os.path.realpath(__file__))
mol_list = os.path.join(path, "data", 'list_water2.txt')
spin_list = os.path.join(path, "data", 'list_water_spins.txt')
charge_list = os.path.join(path, "data", 'list_water_charges.txt')


def test_no_split():
    nameout = tempfile.mktemp()
    sufix = "_alpha_beta.npy"
    rho.main(['--rep', 'atom', '--mol', mol_list, '--spin', spin_list, '--charge', charge_list, '--name', nameout])
    reps = np.load(nameout+sufix)
    assert (reps.shape == (9,414))


def test_split_once():
    nameout = tempfile.mktemp()
    sufix = "_alpha_beta.npy"
    rho.main(['--rep', 'atom', '--mol', mol_list, '--spin', spin_list, '--charge', charge_list, '--name', nameout, '--split'])
    reps = np.load(nameout+sufix, allow_pickle=True)  # why is the `dtype` object ????
    assert (reps.shape == (3, 3, 414))


def test_split_twice():
    nameout = tempfile.mktemp()
    sufix = "_alpha_beta.npy"
    rep_files = [nameout+"_"+os.path.basename(f).split(".")[0]+sufix for f in np.loadtxt(mol_list, dtype=str)]
    rho.main(['--rep', 'atom', '--mol', mol_list, '--spin', spin_list, '--charge', charge_list, '--name', nameout, '--split', "--split"])
    for f in rep_files:
        reps = np.load(f, allow_pickle=True)  # why is the `dtype` object ????
        assert (reps.shape == (3, 414))


if __name__ == '__main__':
    test_no_split()
    test_split_once()
    test_split_twice()
