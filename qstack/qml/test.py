import numpy as np
import ase.io
from slatm import get_mbtypes, get_slatm, get_slatm_for_dataset
from tqdm import tqdm

xyzs = ['4118.xyz']

import os
import random

xyzs = [f'qm7/{f}' for f in sorted(os.listdir("qm7/")) if f[0]!='.']
random.seed(666)
random.shuffle(xyzs)

xyzs = xyzs

#mols = [ase.io.read(xyz) for xyz in xyzs]
#
#mbtypes = get_mbtypes([mol.numbers for mol in mols])
#print(mbtypes[1])
#
#print()
#
#mbtypes = get_mbtypes([mol.numbers for mol in mols], qml=True)
#print(mbtypes[1])
#
##exit(0)
#
#for mol in tqdm(mols):
#    get_slatm(mol.numbers, mol.positions, mbtypes, dgrid3=0.3, dgrid2=0.3)






v0 = get_slatm_for_dataset(xyzs, progress=True) #, dgrid2=0.3, dgrid3=0.3)
np.save('test', v0)

