"""Q-stack."""

import sys
import warnings
import builtins
from qstack import tools
from qstack import constants
from qstack import mathutils


if sys.version_info[1]<10:
    warnings.warn('Redefining built-in function zip for compatibility', stacklevel=1)
    _zip = builtins.zip
    def zip_override(*iterables, strict=False):
        """Override built-in zip for python<3.10 to ignore `strict` argument."""
        return _zip(*iterables)
    builtins.zip = zip_override
