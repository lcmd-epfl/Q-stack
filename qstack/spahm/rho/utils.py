import os
import numpy as np
from types import SimpleNamespace
import qstack.spahm.compute_spahm as spahm
import qstack.spahm.guesses as guesses
from qstack import compound


defaults = SimpleNamespace(
    guess='LB',
    model='Lowdin-long-x',
    basis='minao',
    auxbasis='ccpvdzjkfit',
    omod=['alpha', 'beta'],
    elements=["H", "C", "N", "O", "S"],
    cutoff=5.0,
    xc='hf',
    bpath=os.path.dirname(__file__)+'/basis_opt',
  )


def get_chsp(fname, n):
    if fname:
        chsp = np.loadtxt(fname, dtype=int, ndmin=1)
        if(len(chsp)!=n):
            raise RuntimeError(f'Wrong lengh of the file {fname}')
    else:
        chsp = np.zeros(n, dtype=int)
    return chsp


def load_mols(xyzlist, charge, spin, basis, printlevel=0, units='ANG'):
    mols = []
    for xyzfile, ch, sp in zip(xyzlist, charge, spin):
        if printlevel>0: print(xyzfile, flush=True)
        mols.append(compound.xyz_to_mol(xyzfile, basis, charge=0 if ch is None else ch, spin=0 if ch is None else sp, unit=units)) #TODO
    if printlevel>0: print()
    return mols


def mols_guess(mols, xyzlist, guess, xc=defaults.xc, spin=None, readdm=False, printlevel=0):
    dms = []
    guess = guesses.get_guess(guess)
    for xyzfile, mol in zip(xyzlist, mols):
        if printlevel>0: print(xyzfile, flush=True)
        if not readdm:
            e, v = spahm.get_guess_orbitals(mol, guess, xc=xc)
            dm   = guesses.get_dm(v, mol.nelec, mol.spin if spin else None)
        else:
            dm = np.load(readdm+'/'+os.path.basename(xyzfile)+'.npy')
            if spin and dm.ndim==2:
                dm = np.array((dm/2,dm/2))
        dms.append(dm)
        if printlevel>0: print()
    return dms


def dm_open_mod(dm, omod):
    dmmod = {'sum':   lambda dm: dm[0]+dm[1],
             'diff':  lambda dm: dm[0]-dm[1],
             'alpha': lambda dm: dm[0],
             'beta':  lambda dm: dm[1]}
    return dmmod[omod](dm)


def get_xyzlist(xyzlistfile):
    return np.loadtxt(xyzlistfile, dtype=str, ndmin=1)


def load_reps(f_in, from_list=True, single=True, with_labels=False, local=True, reaction=False):
    if from_list:
        X_list = get_xyzlist(f_in)
        Xs = [np.load(f_X, allow_pickle=True) for f_X in X_list]
    else:
        Xs = [np.load(f_in, allow_pickle=True)]
    reps = []
    for x in Xs:
        labels = []
        if local == True:
            if  type(x[0,0]) == str:
                reps.append(x[:,1])
                labels.append(x[:,0])
            else:
                reps.extend(x)
        else:
           if type(x[0]) == str:
                reps.extend(x[1])
                labels.extend(x[0])
           else:
                reps.extend(x)
    try:
        reps = np.array(reps, dtype=float)
    except:
        raise RuntimeError("Error while loading representations, check the parameters")
    reps = np.array(reps, ndmin=1)
    if with_labels:
        return reps, labels
    else:
        return reps
