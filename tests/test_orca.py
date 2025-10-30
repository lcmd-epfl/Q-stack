#!/usr/bin/env python3

import os
import numpy as np
from pyscf.data import elements
import qstack.orcaio
import qstack.compound
import qstack.fields
from qstack.tools import reorder_ao


def _dipole_moment(mol, dm):
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    mass = np.round(np.array(elements.MASSES)[charges], 3)
    mass_center = np.einsum('i,ix->x', mass, coords) / sum(mass)
    with mol.with_common_orig(mass_center):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    el_dip = np.einsum('xij,ji->x', ao_dip, dm.sum(axis=0))
    nucl_dip = np.einsum('i,ix->x', charges, coords-mass_center)
    mol_dip = nucl_dip - el_dip
    return mol_dip


def test_orca_density_reader():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = qstack.compound.xyz_to_mol(path+'/data/orca/H2O.xyz', 'sto3g', charge=1, spin=1)
    dm = qstack.fields.dm.get_converged_dm(mol, 'HF')
    dm = np.array([dm[0]+dm[1], dm[0]-dm[1]])

    dm400 = qstack.orcaio.read_density(mol, 'H2O.orca400', directory=path+'/data/orca/', version=400, openshell=True)
    dm421 = qstack.orcaio.read_density(mol, 'H2O.orca421', directory=path+'/data/orca/', version=421, openshell=True)
    dm504 = qstack.orcaio.read_density(mol, 'H2O.orca504', directory=path+'/data/orca/', version=504, openshell=True)

    assert(np.linalg.norm(dm-dm400)<1e-4)
    assert(np.linalg.norm(dm400-dm421)<1e-10)
    assert(np.linalg.norm(dm504-dm421)<5e-3)


def test_orca_gbw_reader():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = qstack.compound.xyz_to_mol(path+'/data/orca/H2O.xyz', 'sto3g', charge=1, spin=1)

    mf, _ = qstack.fields.dm.get_converged_mf(mol, 'HF')
    c = mf.mo_coeff
    e = mf.mo_energy
    occ = mf.mo_occ
    def compare_MO(c0, c1):
        for s in range(c0.shape[0]):
            for i in range(c0.shape[-1]):
                assert 5e-3 > min(np.linalg.norm(c0[s,:,i]-c1[s,:,i]), np.linalg.norm(c0[s,:,i] + c1[s,:,i]))

    c504, e504, occ504 = qstack.orcaio.read_gbw(mol, path+'/data/orca/H2O.orca504.gbw')
    compare_MO(c, c504)
    assert np.allclose(e504, e)
    assert np.all(occ504==occ)  # fine since they contain only 0.0 and 1.0

    c421, e421, occ421 = qstack.orcaio.read_gbw(mol, path+'/data/orca/H2O.orca421.gbw')
    compare_MO(c, c421)
    assert np.allclose(e421, e)
    assert np.all(occ421==occ)

    c400, e400, occ400 = qstack.orcaio.read_gbw(mol, path+'/data/orca/H2O.orca400.gbw', sort_l=False)
    compare_MO(c, c400)
    assert np.allclose(e400, e)
    assert np.all(occ400==occ)


def test_orca_gbw_reader_def2tzvp():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = qstack.compound.xyz_to_mol(path+'/data/orca/CEHZOF/CEHZOF.xyz', 'def2tzvp', ecp='def2tzvp')
    c504, _e504, occ504 = qstack.orcaio.read_gbw(mol, path+'/data/orca/CEHZOF/CEHZOF_1_SPE.gbw')
    dm = np.zeros_like(c504)
    for i, (ci, occi) in enumerate(zip(c504, occ504, strict=True)):
        dm[i,:,:] = (ci[:,occi>0] * occi[occi>0]) @ ci[:,occi>0].T
    mol_dip = _dipole_moment(mol, dm)
    mol_dip_true = np.array([-0.98591, -2.20093, 2.61135])
    assert np.linalg.norm(mol_dip-mol_dip_true) < 1e-5


def test_orca_input_reader():
    path = os.path.dirname(os.path.realpath(__file__))
    mol0 = qstack.compound.xyz_to_mol(path+'/data/orca/H2O.xyz', 'sto3g', charge=1, spin=1)
    mol = qstack.orcaio.read_input(path+'/data/orca/H2O.orca504.inp', 'sto3g')
    assert mol.natm == mol0.natm
    assert mol.nelectron == mol0.nelectron
    assert np.all(mol.elements==mol0.elements)
    assert np.allclose(mol.atom_coords(), mol0.atom_coords())


def test_orca_density_reader_def2tzvp():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = qstack.compound.xyz_to_mol(path+'/data/orca/CEHZOF/CEHZOF.xyz', 'def2tzvp', ecp='def2tzvp')
    c, _e, occ = qstack.orcaio.read_gbw(mol, path+'/data/orca/CEHZOF/CEHZOF_1_SPE.gbw')
    c = c.squeeze()
    occ = occ.squeeze()
    dm0 = (c[:,occ>0] * occ[occ>0]) @ c[:,occ>0].T

    dm = qstack.orcaio.read_density(mol, 'CEHZOF_1_SPE', directory=path+'/data/orca/CEHZOF', version=504, reorder_dest=None)
    Co_idx = [mol.atom_symbol(i) for i in range(mol.natm)].index('Co')
    ls_from_orca = {Co_idx : [0]*6 + [1]*3 + [2]*3 + [1, 2, 3]}
    idx = qstack.orcaio._get_indices(mol, ls_from_orca)
    dm = dm[np.ix_(idx,idx)]
    dm = reorder_ao(mol, dm, src='orca', dest='pyscf')
    assert np.linalg.norm(dm-dm0) < 1e-14


if __name__ == '__main__':
    test_orca_input_reader()
    test_orca_density_reader()
    test_orca_gbw_reader()
    test_orca_gbw_reader_def2tzvp()
    test_orca_density_reader_def2tzvp()
