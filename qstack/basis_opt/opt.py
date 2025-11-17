"""Basis set optimization routines and command-line interface."""

import sys
from ast import literal_eval
import argparse
import numpy as np
import scipy.optimize
from pyscf import gto
import pyscf.data
from ..compound import basis_flatten
from . import basis_tools as qbbt


def optimize_basis(elements_in, basis_in, molecules_in, gtol_in=1e-7, method_in="CG", printlvl=2, check=False):
    """Optimize a given basis set.

    Args:
        elements_in (str): List of elements to optimize. If None, optimize all elements in the basis.
        basis_in (list): List of files paths (str) or dicts containing basis set(s).
        molecules_in (list): List of file paths (str) or dicts containing molecular data.
        gtol_in (float): Gradient norm must be less than gtol_in before successful termination (minimization).
        method_in (str): Type of solver. Check scipy.optimize.minimize for full documentation.
        printlvl (int): Level of printing during optimization (0: none, 1: final basis, 2: detailed).
        check (bool): If True, compute and return both analytical and numerical gradients without optimization.

    Returns:
        Dictionary containing the optimized basis.

    """
    def energy(x):
        """Compute total loss function (fitting error) for given exponents.

        Args:
            x (numpy.ndarray): Log of exponents.

        Returns:
            float: Loss function value.
        """
        exponents = np.exp(x)
        newbasis = qbbt.exp2basis(exponents, myelements, basis)
        E = 0.0
        for m in moldata:
            E += qbbt.energy_mol(newbasis, m)
        return E


    def gradient(x):
        """Compute total loss function (fitting error) and gradient for given exponents.

        Args:
            x (numpy.ndarray): Log of exponents.

        Returns:
            tuple: A tuple containing:
            - E (float): Loss function value.
            - dE_dx (numpy.ndarray): Gradient with respect to log(exponents).
        """
        exponents = np.exp(x)
        newbasis = qbbt.exp2basis(exponents, myelements, basis)

        E = 0.0
        dE_da = np.zeros(nexp)
        for m in moldata:
            E_, dE_da_ = qbbt.gradient_mol(nexp, newbasis, m)
            E     += E_
            dE_da += dE_da_
            if printlvl>=2:
                print('e =', E_, '(', E_/m['self']*100.0, '%)')
        if printlvl>=2:
            print(E, max(abs(dE_da)))
        dE_da = qbbt.cut_myelements(dE_da, myelements, bf_bounds)

        if printlvl>=2:
            print(flush=True)

        dE_dx = dE_da * exponents
        return E, dE_dx


    def gradient_only(x):
        """Compute only the gradient of the loss function (wrapper for optimization algorithms).

        Args:
            x (numpy.ndarray): Log of exponents.

        Returns:
            numpy.ndarray: Gradient with respect to log(exponents).
        """
        return gradient(x)[1]


    def read_bases(basis_files):
        """Read basis set definitions from files or dicts.

        Args:
            basis_files (list): List of file paths (str) or basis dicts.

        Returns:
            dict: Combined basis set definition.

        Raises:
            RuntimeError: If multiple sets for the same element are provided.
        """
        basis = {}
        for i in basis_files:
            if isinstance(i, str):
                with open(i) as f:
                    addbasis = literal_eval(f.read())
                q = list(addbasis.keys())[0]
                if q in basis:
                    raise RuntimeError('several sets for element ' + q)
                basis.update(addbasis)
            else:
                q = list(i.keys())[0]
                if q in basis:
                    raise RuntimeError('several sets for element ' + q)
                basis.update(i)
        return basis


    def make_bf_start():
        """Create basis function index bounds for each element.

        Returns:
            dict: Dictionary mapping elements to their [start, end] indices.
        """
        nbf = [len(basis[q]) for q in elements]
        bf_bounds = {}
        for i, q in enumerate(elements):
            start = sum(nbf[0:i])
            bf_bounds[q] = [start, start+nbf[i]]
        return bf_bounds


    def make_moldata(fname):
        """Create molecular data dictionary from file or dict.

        Args:
            fname (str or dict): Path to .npz file or dictionary containing molecular structure,
                grid coordinates and weights, and reference density evaluated on it.

        Returns:
            dict: Dictionary containing:
            mol (pyscf Mole): pyscf Mole object.
            rho (numpy.ndarray): Reference density values on the grid.
            coords (numpy.ndarray): Grid coordinates.
            weights (numpy.ndarray): Grid weights.
            self (float): Integral of the squared reference density.
            idx (numpy.ndarray): Basis function indices for each AO.
            centers (list): Atomic center indices for each AO.
            distances (numpy.ndarray): Squared distances from each atom to each grid point.
        """
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

        centers, l, _ = basis_flatten(mol, return_both=False)
        idx = np.zeros_like(centers)
        i = 0
        while i < mol.nao:
            q = mol.atom_symbol(centers[i])
            for ib in range(*bf_bounds[q]):
                msize = 2*l[i]+1
                idx[i:i+msize] = [ib] * msize
                i += msize

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
            'distances': distances,
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

    moldata = [make_moldata(fname) for fname in molecules_in]

    if printlvl>=2:
        print("Initial exponents")
        for l, a in zip(angular_momenta, exponents, strict=True):
            print(f'{l=} {a=}')
        print(flush=True)

    x0 = np.log(exponents)
    x1 = qbbt.cut_myelements(x0, myelements, bf_bounds)
    angular_momenta = qbbt.cut_myelements(angular_momenta, myelements, bf_bounds)

    if check:
        gr_num = scipy.optimize.approx_fprime(x1, energy, 1e-4)
        gr_an  = gradient_only(x1)
        return {'num': gr_num, 'an': gr_an, 'diff': gr_an-gr_num}

    xopt = scipy.optimize.minimize(energy, x1, method=method_in, jac=gradient_only,
                                   options={'gtol': gtol_in, 'disp': printlvl}).x

    exponents = np.exp(xopt)
    newbasis = qbbt.exp2basis(exponents, myelements, basis)
    if printlvl>=1:
        qbbt.printbasis(newbasis, sys.stdout)

    return newbasis


def _get_arg_parser():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description='Optimize a density fitting basis set.')
    parser.add_argument('--elements',  type=str,   dest='elements',  nargs='+',    help='elements for optimization')
    parser.add_argument('--basis',     type=str,   dest='basis',     nargs='+',    help='initial df bases', required=True)
    parser.add_argument('--molecules', type=str,   dest='molecules', nargs='+',    help='molecules', required=True)
    parser.add_argument('--gtol',      type=float, dest='gtol',      default=1e-7, help='tolerance')
    parser.add_argument('--method',    type=str,   dest='method',    default='CG', help='minimization algoritm')
    parser.add_argument('--print',     type=int,   dest='print',     default=2,    help='printing level')
    parser.add_argument('--check', action='store_true', dest='check', default=False, help='check the gradient and exit')
    return parser


def main():
    """Run basis set optimization via command-line interface."""
    args = _get_arg_parser().parse_args()

    result = optimize_basis(args.elements, args.basis, args.molecules, args.gtol, args.method, check=args.check, printlvl=args.print)
    if args.check is False:
        if args.print==0:
            qbbt.printbasis(result, sys.stdout)
    else:
        gr_an, gr_num, gr_diff = result['an'], result['num'], result['diff']
        print('analytical gradient')
        print(gr_an)
        print('numerical gradient')
        print(gr_num)
        print('difference')
        print(gr_diff)
        print('relative difference')
        print(gr_diff/gr_num)
        print(flush=True)


if __name__ == "__main__":
    main()
