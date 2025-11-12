import numpy as np
from qstack import compound
from .compute_rho_spahm import get_repr
from .parser import SpahmParser


def main(args=None):
    parser = SpahmParser(description='This program computes the SPAHM(a) representation for a given molecular system', atom=True)
    parser.add_argument('--mol',       dest='mol',           required=True,                        type=str, help="the path to the xyz file with the molecular structure")
    parser.add_argument('--charge',    dest='charge',        default=0,                            type=int, help='total charge of the system (default: 0)')
    parser.add_argument('--spin',      dest='spin',          default=None,                         type=int, help='number of unpaired electrons (default: None) (use 0 to treat a closed-shell system in a UHF manner)')
    args = parser.parse_args(args=args)
    if args.print>0:
        print(vars(args))

    mol = compound.xyz_to_mol(args.mol, args.basis, charge=args.charge, spin=args.spin, unit=args.units, ecp=args.ecp)

    if args.elements is None:
        elements = sorted(mol.elements)
    else:
        elements = args.elements

    representations = get_repr(rep_type="atom", mols=[mol], xyzlist=[args.mol], guess=args.guess,
                               xc=args.xc, spin=args.spin, readdm=args.readdm,
                               omods=args.omod, elements=elements,
                               auxbasis=args.auxbasis, model=args.model,
                               only_z=args.only_z,
                               split=args.split, with_symbols=args.with_symbols, merge=args.merge,
                               printlevel=args.print)
    mol_name = args.mol.split('/')[-1].split('.')[0]
    if args.name_out is not None:
        name_out = args.name_out
    else:
        name_out = 'X_'+mol_name
    if args.spin is not None:
        name_out = name_out+'_'+'_'.join(args.omod)
    np.save(name_out, representations)


if __name__ == '__main__':
    main()
