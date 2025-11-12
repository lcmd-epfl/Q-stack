import os
from types import SimpleNamespace
from qstack.tools import FrozenKeysDict


class OmodFnsDict(FrozenKeysDict):
    def __init__(self, dictionary=None):
        _omod_fns_names = ('alpha', 'beta', 'sum', 'diff')
        super().__init__(_omod_fns_names, dictionary)


class ModelsDict(FrozenKeysDict):
    def __init__(self, dictionary=None):
        _omod_fns_names = ('pure', 'sad-diff', 'occup', 'lowdin-short', 'lowdin-long', 'lowdin-short-x', 'lowdin-long-x', 'mr2021')
        super().__init__(_omod_fns_names, dictionary)


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
