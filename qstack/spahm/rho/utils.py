import os
import warnings
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
            raise RuntimeError(f'Wrong length of the file {fname}')
    else:
        chsp = np.zeros(n, dtype=int)
    return chsp


def load_mols(xyzlist, charge, spin, basis, printlevel=0, units='ANG', ecp=None, progress=False):
    mols = []
    if progress:
        import tqdm
        xyzlist = tqdm.tqdm(xyzlist)
    for xyzfile, ch, sp in zip(xyzlist, charge, spin):
        if printlevel>0: print(xyzfile, flush=True)
        mols.append(compound.xyz_to_mol(xyzfile, basis,
                                        charge=0 if ch is None else ch,
                                        spin=0 if sp is None else sp,
                                        unit=units, ecp=ecp))
    if printlevel>0: print()
    return mols


def mols_guess(mols, xyzlist, guess, xc=defaults.xc, spin=None, readdm=False, printlevel=0):
    dms = []
    guess = guesses.get_guess(guess)
    for xyzfile, mol in zip(xyzlist, mols):
        if printlevel>0: print(xyzfile, flush=True)
        if not readdm:
            e, v = spahm.get_guess_orbitals(mol, guess, xc=xc)
            dm   = guesses.get_dm(v, mol.nelec, mol.spin if spin is not None else None)
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

def check_data_struct(fin, local=False):
    x = np.load(fin, allow_pickle=True)
    if type(x.flatten()[0]) == str or type(x.flatten()[0]) == np.str_:
        is_labeled = True
        if not local and x.shape[0] == 1:
            is_single = True
        elif  x.shape[1] == 2:
            is_single = True
        else:
            is_single = False
    else:
        is_labeled = False
        if not local and x.ndim == 1:
            is_single = True
        elif x.shape[1] != 2: ## could be problematic! (if it's a set of local representations and nfeatures = 2 !!
            is_single=True
        else:
            is_single = False
    return is_single, is_labeled



def load_reps(f_in, from_list=True, srcdir=None, with_labels=False,
              local=True, sum_local=False, printlevel=0, progress=False,
              file_format={'is_labeled':None, 'is_single':None}):
    '''
    A function to load representations from txt-list/npy files.
        Args:
            - f_in: the name of the input file
            - from_list(bool): if the input file is a txt-file containing a list of paths to the representations
            - srcdir(str) : the path prefix to be at the begining of each file in `f_in`, defaults to cwd
            - with_label(bool): saves a list of tuple (filename, representation)
            - local(bool): if the representations is local
            - sum_local(bool): if local=True then sums the local components
            - printlevel(int): level of verbosity
            - progress(bool): if True shows progress-bar
            - file_format(dict): (for "experienced users" only) structure of the input data, defaults to structure auto determination
        Return:
            np.array with shape (N,M) where N number of representations M dimmensionality of the representation
            OR tuple ([N],np.array(N,M)) containing list of labels and np.array of representations
    '''
    if srcdir == None:
        path2list = os.getcwd()
    else:
        path2list = srcdir
    if from_list:
        X_list = get_xyzlist(f_in)
        Xs = []     # Xs must be a list of representations (local or global) whose len() = # of representations
        for f_X in X_list:
            if None in file_format.values():
                is_single, is_labeled = check_data_struct(os.path.join(path2list,f_X), local=local)
            else:
                is_single, is_labeled = file_format['is_single'], file_format['is_labeled']
            if printlevel > 0 : print(is_single, is_labeled)
            Xs.append(np.load(os.path.join(path2list,f_X), allow_pickle=True))
    else:
        if None in file_format.values():
            is_single, is_labeled = check_data_struct(os.path.join(path2list,f_in), local=local)
        else:
            is_single, is_labeled = file_format['is_single'], file_format['is_labeled']
        # if the given file contains a single representation create a one-element list
        Xs = [np.load(f_in, allow_pickle=True)] if is_single else np.load(f_in, allow_pickle=True)
    print(f"Loading {len(Xs)} representations (local = {local}, labeled = {is_labeled})")
    if progress:
        import tqdm
        Xs = tqdm.tqdm(Xs)
    reps = []
    labels = []
    for x in Xs:
        if local == True:
            if is_labeled:
                if sum_local:
                    reps.append(x[:,1].sum(axis=0))
                else:
                    reps.extend(x[:,1])
                    labels.extend(x[:,0])
            else:
                reps.extend(x)
        else:
           if is_labeled:
                reps.append(x[1])
                labels.extend(x[0])
           else:
                reps.append(x)
    try:
        reps = np.array(reps, dtype=float)
    except:
        print(len(reps))
        shapes = [r.shape[0]  for r in reps]
        shapes = set(shapes)
        print(shapes)
        raise RuntimeError(f"Error while loading representations in {f_in}, check the parameters")
    if with_labels and (len(labels) < len(reps)):
        warnings.warn("All labels could not be recovered (verify your representation files).", RuntimeWarning)
    if with_labels:
        return reps, labels
    else:
        return reps

def regroup_symbols(file_list, print_level=0):
    reps, atoms = load_reps(file_list, from_list=True, with_labels=True, local=True, printlevel=print_level)
    if print_level > 0: print(f"Extracting {len(atoms)} atoms from {file_list}:")
    atoms_set = {e:[] for e in set(atoms)}
    for e, v in zip(atoms, reps):
        atoms_set[e].append(v)
    if print_level > 0: print([(k, len(v)) for k, v in atoms_set.items()])
    return atoms_set
