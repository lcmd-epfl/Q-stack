import os
import glob
from types import SimpleNamespace
import numpy as np
import ase.io
from qstack.qml.slatm import get_slatm_rxn
from qstack.qml.b2r2 import get_b2r2


class Rxn_data:
    def __init__(self, data_dir='.'):
        def get_gdb_xyz_files(idx):
            r = [f'{data_dir}/xyz/r{idx:06}.xyz']
            p = sorted(glob.glob(f'{data_dir}/xyz/p{idx:06}*.xyz'))
            return r, p
        self.get_gdb7_data = self.get_data_template(idx_path=f'{data_dir}/idx.dat',
                                                    input_bohr=True,
                                                    get_xyz_files=get_gdb_xyz_files)

    def get_data_template(self, idx_path, input_bohr=False, get_xyz_files=None):
        def read_mols(files):
            sub_mols = []
            for f in files:
                mol = ase.io.read(f)
                if input_bohr:
                    mol.set_positions(mol.positions*ase.units.Bohr)
                sub_mols.append(mol)
            return sub_mols
        def get_data():
            indices = np.loadtxt(idx_path, dtype=int)
            reactions = []
            for idx in indices:
                rfiles, pfiles = get_xyz_files(idx)
                reactions.append(SimpleNamespace(reactants=read_mols(rfiles),
                                                 products=read_mols(pfiles)))
            return reactions
        return get_data


def test_b2r2_l():
    _test_b2r2('l')
def test_b2r2_a():
    _test_b2r2('a')
def test_b2r2_n():
    _test_b2r2('n')


def _test_b2r2(variant):
    data_dir = f'{os.path.dirname(os.path.realpath(__file__))}/data/rxn-repr'
    reactions = Rxn_data(data_dir=data_dir).get_gdb7_data()
    b2r2 = get_b2r2(reactions, variant=variant)
    b2r2_0 = np.load(f'{data_dir}/b2r2_{variant}.npy')
    assert(np.linalg.norm(b2r2-b2r2_0) < 1e-10)


def test_slatm_rxn():
    data_dir = f'{os.path.dirname(os.path.realpath(__file__))}/data/rxn-repr'
    reactions = Rxn_data(data_dir=data_dir).get_gdb7_data()
    slatm = get_slatm_rxn(reactions, qml_mbtypes=True, progress=False)
    slatm_0 = np.load(f'{data_dir}/slatm_d.npy')
    assert(np.linalg.norm(slatm-slatm_0) < 1e-10)


if __name__ == '__main__':
    test_slatm_rxn()
    test_b2r2_l()
    test_b2r2_a()
    test_b2r2_n()
