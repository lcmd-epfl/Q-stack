import os
import sys
import numpy as np
import scipy.optimize
from pyscf import gto
import pyscf.data
from qstack.basis_opt import basis_tools as qbbt


def optimize_basis(elements_in, basis_in, molecules_in, gtol_in=1e-7, method_in="CG", check=False):

    def energy(x):
        exponents = np.exp(x)
        newbasis = qbbt.exp2basis(exponents, myelements, basis)
        E = 0.0
        for m in moldata:
            E += qbbt.energy_mol(newbasis, m)
        return E

    def gradient(x):
        exponents = np.exp(x)
        newbasis = qbbt.exp2basis(exponents, myelements, basis)

        E = 0.0
        dE_da = np.zeros(nexp)
        for m in moldata:
            E_, dE_da_ = qbbt.gradient_mol(nexp, bf_bounds, newbasis, m)
            E     += E_
            dE_da += dE_da_
            print('e =', E_, '(', E_/m['self']*100.0, '%)')
        print(E, max(abs(dE_da)))
        dE_da = qbbt.cut_myelements(dE_da, myelements, bf_bounds)

        print(flush=True)

        dE_dx = dE_da * exponents
        return E, dE_dx

    def gradient_only(x):
        return gradient(x)[1]

    def read_bases(basis_files):
        basis = {}
        for i in basis_files:
            if isinstance(i, str):
                with open(i, "r") as f:
                    addbasis = eval(f.read())
                q = list(addbasis.keys())[0]
                if q in basis.keys():
                    print('error: several sets for element', q)
                    exit()
                basis.update(addbasis)
            else:
                q = list(i.keys())[0]
                if q in basis.keys():
                    print('error: several sets for element', q)
                    exit()
                basis.update(i)
        return basis

    def make_bf_start():
        nbf = []
        for q in elements:
            nbf.append(len(basis[q]))
        bf_bounds = {}
        for i, q in enumerate(elements):
            start = sum(nbf[0:i])
            bf_bounds[q] = [start, start+nbf[i]]
        return bf_bounds

    def make_moldata(fname):
        if isinstance(fname, str):
            rho_data = np.load(fname)
        else:
            rho_data = fname

        molecule = rho_data['atom'   ]
        rho      = rho_data['rho'    ]
        coords   = rho_data['coords' ]
        weights  = rho_data['weights']
        self = np.einsum('p,p,p->', weights, rho, rho)
        mol = gto.M(atom=str(molecule), basis=basis)

        idx = []
        centers = []
        for iat in range(mol.natm):
            q = mol._atom[iat][0]
            ib0 = bf_bounds[q][0]
            for ib, b in enumerate(mol._basis[q]):
                l = b[0]
                idx += [ib+ib0] * (2*l+1)
                centers += [iat] * (2*l+1)
        idx = np.array(idx)

        distances = np.zeros((mol.natm, len(rho)))
        for iat in range(mol.natm):
            center = mol.atom_coord(iat)
            distances[iat] = np.sum((coords - center)**2, axis=1)

        return {
            'mol'      : mol,
            'rho'      : rho,
            'coords'   : coords,
            'weights'  : weights,
            'self'     : self,
            'idx'      : idx,
            'centers'  : centers,
            'distances': distances
        }

    basis = read_bases(basis_in)

    elements = sorted(basis.keys(), key=pyscf.data.elements.charge)
    if elements_in:
        myelements = elements_in
        myelements.sort(key=pyscf.data.elements.charge)
    else:
        myelements = elements

    basis_list = [i for q in elements for i in basis[q]]
    angular_momenta = np.array([i[0] for i in basis_list])
    exponents = np.array([i[1][0] for i in basis_list])

    nexp = len(basis_list)
    bf_bounds = make_bf_start()

    moldata = []
    for fname in molecules_in:
        moldata.append(make_moldata(fname))

    print("Initial exponents")
    for l, a in zip(angular_momenta, exponents):
        print('l =', l, 'a = ', a)
    print()

    x0 = np.log(exponents)
    x1 = qbbt.cut_myelements(x0, myelements, bf_bounds)
    angular_momenta = qbbt.cut_myelements(angular_momenta, myelements, bf_bounds)

    if check:
        gr1 = scipy.optimize.approx_fprime(x1, energy, 1e-4)
        gr2 = gradient_only(x1)
        print()
        print('anal')
        print(gr2)
        print('num')
        print(gr1)
        print('diff')
        print(gr1-gr2)
        print('rel diff')
        print((gr1-gr2)/gr1)
        print()
        return None

    xopt = scipy.optimize.minimize(energy, x1, method=method_in, jac=gradient_only,
                                   options={'gtol': gtol_in, 'disp': True}).x

    exponents = np.exp(xopt)
    newbasis = qbbt.exp2basis(exponents, myelements, basis)
    qbbt.printbasis(newbasis, sys.stdout)

    return newbasis

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Optimize a density fitting basis set.')
    parser.add_argument('--elements',  type=str,   dest='elements',  nargs='+',    help='elements for optimization')
    parser.add_argument('--basis',     type=str,   dest='basis',     nargs='+',    help='initial df bases', required=True)
    parser.add_argument('--molecules', type=str,   dest='molecules', nargs='+',    help='molecules', required=True)
    parser.add_argument('--gtol',      type=float, dest='gtol',      default=1e-7, help='tolerance')
    parser.add_argument('--method',    type=str,   dest='method',    default='CG', help='minimization algoritm')
    parser.add_argument('--check', action='store_true', dest='check', default=False, help='check the gradient and exit')
    args = parser.parse_args()

    ret = optimize_basis(args.elements, args.basis, args.molecules, args.gtol, args.method, check=args.check)


if __name__ == "__main__":
    main()
