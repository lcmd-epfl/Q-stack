#!/usr/bin/env python3

import os
import polars as pl
from lcmd_db import load_dataset


def test_loading_dataset():
    data = load_dataset("oscar_nhc")
    molecules = data.as_dataset("molecules")
    assert (len(molecules) == 8622)
    return None


def test_loading_molecule():
    data = load_dataset("oscar_nhc")
    molecules = data.as_dataset("molecules")
    mol = molecules.filter(pl.col("id") == 178743)[0]
    mol_smiles = "[C]1SN=C2COC[C@H](c3ccccc3)N12"
    assert (mol_smiles==mol.properties["smiles"])
    return None


def test_molecule_query():
    aromatic_smarts = "c1ccccc1"
    tot_candidates = 7896
    candidate_data = load_dataset("oscar_nhc", include=["molecules"], smarts=aromatic_smarts)
    candidate_mols =  candidate_data.as_dataset("molecules")
    assert (len(candidate_mols) == tot_candidates)
    smiles = [mol.properties["smiles"] for mol in candidate_mols]
    path = os.path.dirname(os.path.realpath(__file__))
    with open(path+"/data/SMARTS_query-oscar_nhc-c1ccccc1.txt") as fout:
        true_smiles = [l.strip("\n") for l in fout]
    for smi in smiles:
        assert (smi in true_smiles)
    return None


if __name__ == '__main__':
    test_loading_molecule()
    test_loading_dataset()
    test_molecule_query()
