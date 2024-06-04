"""Q-stack."""

from qstack import tools
from qstack import compound
from qstack import constants
from qstack import fields
from qstack import basis_opt
from qstack import spahm
from qstack import mathutils
from qstack import orcaio

try:
    import sklearn
except ImportError:
    pass
else:
    from qstack import regression
    del sklearn
