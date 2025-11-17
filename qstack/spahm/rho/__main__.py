"""Command-line entry point for SPAHM(a,b) computation.

Main executable script for ρ-SPAHM, also found in ``qstack.spahm.rho.compute_rho_spahm``
"""

from .compute_rho_spahm import main
from . import compute_rho_spahm


def _get_arg_parser():
    # this re-def is just to make the docs work properly
    return compute_rho_spahm._get_arg_parser()


if __name__ == "__main__":
    main()
