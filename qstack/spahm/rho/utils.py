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


def load_mols(xyzlist, charge, spin, basis, printlevel=0, units='ANG', ecp=None):
    mols = []
    if printlevel > 0:
        progress = add_progressbar(max_value=len(xyzlist))
        i = 0
    for xyzfile, ch, sp in zip(xyzlist, charge, spin):
        if printlevel>1: print(xyzfile, flush=True)
        mols.append(compound.xyz_to_mol(xyzfile, basis,
                                        charge=0 if ch is None else ch,
                                        spin=0 if sp is None else sp,
                                        unit=units, ecp=ecp))
        if printlevel > 0:
            i+=1
            progress.update(i)
    if printlevel>0:
        progress.finish()
        print()
    if printlevel>1: print()
    return mols


def mols_guess(mols, xyzlist, guess, xc=defaults.xc, spin=None, readdm=False, printlevel=0):
    dms = []
    guess = guesses.get_guess(guess)
    for xyzfile, mol in zip(xyzlist, mols):
        if printlevel>0: print(xyzfile, flush=True)
        if not readdm:
            e, v = spahm.get_guess_orbitals(mol, guess, xc=xc)
            dm   = guesses.get_dm(v, mol.nelec, mol.spin if spin != None else None)
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


def load_reps(f_in, from_list=True, single=False, with_labels=False, local=True, summ=False, printlevel=0):
    '''
    A function to load representations from txt-list/npy files.
        Args:
            - f_in: the name of the input file
            - fomr_list(bol): if the input file is a txt-file containing a list of paths to the representations
            - single: if the input file is .npy AND contains a single representation (not an array of representation)
            - with_label(bol): saves a list of tuple (filename, representation)
            - local(bol): if the representations is local
            - summ(bol): if local=True then summs the local components
            - printlevel(int): level of verbosity
        Return:
            np.array with shape (N,M) where N number of representations M dimmensionality of the representation
            OR tuple (N,np.array(N,M)) containing filenames in pos 0
    '''
    if from_list:
        X_list = get_xyzlist(f_in)
        if printlevel > 0:
            progress = add_progressbar(max_value=len(X_list)*2)
            i=0
        Xs = []
        for f_X in X_list:
            Xs.append(np.load(f_X, allow_pickle=True))
            if printlevel > 0:
                i+=1
                progress.update(i)
    else:
        Xs = np.load(f_in, allow_pickle=True) if not single else np.array([np.load(f_in, allow_pickle=True)])
        if printlevel > 0:
            progress = add_progressbar(max_value=len(Xs))
            i=0
    reps = []
    labels = []
    for x in Xs:
        if local == True:
            if x.ndim > 1 and (type(x[0,0]) == str or type(x[0,0]) == np.str_):
                if summ:
                    reps.append(x[:,1].sum(axis=0))
                else:
                    reps.extend(x[:,1])
                    labels.extend(x[:,0])
            else:
                reps.extend(x)
        else:
           if type(x[0]) == str or type(x[0]) == np.str_:
                reps.append(x[1])
                labels.extend(x[0])
           else:
                reps.append(x) 
        if printlevel > 0:
            i+=1
            progress.update(i)
    if printlevel > 0: progress.finish()
    try:
        reps = np.array(reps, dtype=float)
    except:
        print(len(reps))
        shapes = [r.shape[0]  for r in reps]
        shapes = set(shapes)
        print(shapes)
        raise RuntimeError("Error while loading representations, check the parameters")
    if with_labels:
        return reps, labels
    else:
        return reps

def add_progressbar(legend='', max_value=100):
    import progressbar
    import time
    #import logging
    #progressbar.streams.wrap_stderr()
    #logging.basicConfig()
    widgets=[\
    ' [', progressbar.Timer(), '] ',\
    progressbar.Bar(),\
    ' (', progressbar.ETA(), ')']
    bar = progressbar.ProgressBar(widgets=widgets, max_value=max_value, redirect_stdout=True).start()
    return bar

def build_reaction(reacts_file, prods_file, local=False, print_level=0, summ=True, diff=True):
    reacts = []
    with open(reacts_file, 'r') as r_in:
        lines = r_in.readlines()
        for line in lines:
            line = line.rstrip('\n')
            structs = line.split(' ')
            reacts.append(structs)
    prods = []
    with open(prods_file, 'r') as p_in:
        lines = p_in.readlines()
        for line in lines:
            line = line.rstrip('\n')
            structs = line.split(' ')
            prods.append(structs)
    tot = len(reacts)+len(prods)
    if print_level > 0 : progress = add_progressbar(max_value=tot)
    i = 0
    XR = []
    for rxn in reacts:
        xr = []
        for r in rxn:
            xr.append(load_reps(r, from_list=False, with_labels=False, local=local, summ=True if local else False, single=True))
        xr = np.squeeze(xr)
#        print(xr.shape)
#        exit()
        if summ and xr.ndim > 1:
            xr = xr.sum(axis=0)
        XR.append(xr)
        i+=1
        if print_level > 0 : progress.update(i)
    XP = []
    for rxn in prods:
        xp=[]
        for p in rxn:
            xp.append(load_reps(p, from_list=False, with_labels=False, local=local, summ=True if local else False, single=True))
        xp = np.squeeze(xp)
        if summ and xp.ndim > 1:
            xp = xp.sum(axis=0)
        XP.append(xp)
        i+=1
        if print_level > 0 : progress.update(i)
    XR = np.squeeze(XR)
    XP = np.squeeze(XP)
    if diff : rxn = XP - XR
    else: rxn = (XR, XP)
    return rxn

def regroup_symbols(file_list, print_level=0):
    reps, atoms = load_reps(file_list, from_list=True, with_labels=True, local=True, printlevel=print_level)
    if print_level > 0: print(f"Extracting {len(atoms)} atoms from {file_list}:")
    atoms_set = {e:[] for e in set(atoms)}
    for e, v in zip(atoms, reps):
        atoms_set[e].append(v)
    if print_level > 0: print([(k, len(v)) for k, v in atoms_set.items()])
    return atoms_set


