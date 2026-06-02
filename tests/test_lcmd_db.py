#!/usr/bin/env python3

from lcmd_db import load_dataset
import polars as pl


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


if __name__ == '__main__':
    test_loading_molecule()
