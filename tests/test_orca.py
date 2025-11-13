#!/usr/bin/env python3

import os
import numpy as np
from pyscf.data import elements
from qstack import orcaio, compound, fields


def _dipole_moment(mol, dm):
    coords = mol.atom_coords()
    mass = np.array(elements.MASSES)[qstack.compound.numbers(mol)]
    mass_center = np.einsum('i,ix->x', mass, coords) / sum(mass)
    with mol.with_common_orig(mass_center):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    el_dip = np.einsum('xij,ji->x', ao_dip, dm)
    nucl_dip = np.einsum('i,ix->x', mol.atom_charges(), coords-mass_center)
    mol_dip = nucl_dip - el_dip
    return mol_dip


def test_orca_density_reader():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/orca/H2O.xyz', 'sto3g', charge=1, spin=1)
    dm = fields.dm.get_converged_dm(mol, 'HF')
    dm = np.array([dm[0]+dm[1], dm[0]-dm[1]])

    dm400 = orcaio.read_density(mol, 'H2O.orca400', directory=path+'/data/orca/', version=400, openshell=True)
    dm421 = orcaio.read_density(mol, 'H2O.orca421', directory=path+'/data/orca/', version=421, openshell=True)
    dm504 = orcaio.read_density(mol, 'H2O.orca504', directory=path+'/data/orca/', version=504, openshell=True)

    assert (np.linalg.norm(dm-dm400)<1e-4)
    assert (np.linalg.norm(dm400-dm421)<1e-10)
    assert (np.linalg.norm(dm504-dm421)<5e-3)


def test_orca_gbw_reader():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/orca/H2O.xyz', 'sto3g', charge=1, spin=1)

    mf, _ = fields.dm.get_converged_mf(mol, 'HF')
    c = mf.mo_coeff
    e = mf.mo_energy
    occ = mf.mo_occ

    def compare_MO(c0, c1):
        for s in range(c0.shape[0]):
            for i in range(c0.shape[-1]):
                assert 5e-3 > min(np.linalg.norm(c0[s,:,i]-c1[s,:,i]), np.linalg.norm(c0[s,:,i] + c1[s,:,i]))

    c504, e504, occ504 = orcaio.read_gbw(mol, path+'/data/orca/H2O.orca504.gbw')
    compare_MO(c, c504)
    assert np.allclose(e504, e)
    assert np.all(occ504==occ)  # fine since they contain only 0.0 and 1.0

    c421, e421, occ421 = orcaio.read_gbw(mol, path+'/data/orca/H2O.orca421.gbw')
    compare_MO(c, c421)
    assert np.allclose(e421, e)
    assert np.all(occ421==occ)

    c400, e400, occ400 = orcaio.read_gbw(mol, path+'/data/orca/H2O.orca400.gbw', sort_l=False)
    compare_MO(c, c400)
    assert np.allclose(e400, e)
    assert np.all(occ400==occ)


def test_orca_gbw_reader_def2tzvp():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/orca/CEHZOF/CEHZOF.xyz', 'def2tzvp', ecp='def2tzvp')
    c504, _e504, occ504 = orcaio.read_gbw(mol, path+'/data/orca/CEHZOF/CEHZOF_1_SPE.gbw')
    dm = fields.dm.make_rdm1(c504[0], occ504[0])
    mol_dip = _dipole_moment(mol, dm)
    mol_dip_true = np.array([-0.98591, -2.20093, 2.61135])
    assert np.linalg.norm(mol_dip-mol_dip_true) < 1e-5


def test_orca_input_reader():
    path = os.path.dirname(os.path.realpath(__file__))
    mol0 = compound.xyz_to_mol(path+'/data/orca/H2O.xyz', 'sto3g', charge=1, spin=1)
    mol = orcaio.read_input(path+'/data/orca/H2O.orca504.inp', 'sto3g')
    assert mol.natm == mol0.natm
    assert mol.nelectron == mol0.nelectron
    assert np.all(mol.elements==mol0.elements)
    assert np.allclose(mol.atom_coords(), mol0.atom_coords())


def test_orca_density_reader_def2tzvp():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/orca/CEHZOF/CEHZOF.xyz', 'def2tzvp', ecp='def2tzvp')
    c, _e, occ = orcaio.read_gbw(mol, path+'/data/orca/CEHZOF/CEHZOF_1_SPE.gbw')
    dm0 = fields.dm.make_rdm1(c[0], occ[0])
    dm = orcaio.read_density(mol, 'CEHZOF_1_SPE', directory=path+'/data/orca/CEHZOF', version=504, reorder_dest='pyscf')
    assert np.linalg.norm(dm-dm0) < 1e-14


if __name__ == '__main__':
    test_orca_input_reader()
    test_orca_density_reader()
    test_orca_gbw_reader()
    test_orca_gbw_reader_def2tzvp()
    test_orca_density_reader_def2tzvp()
