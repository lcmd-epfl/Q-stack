# qstack.qml is in a different python package
# but prefer "the local version of it" if we are in a development environment, and both sources are there.
import os
_qstack_qml_path = os.path.join(os.path.dirname(__file__), 'qstack-qml')
if os.path.isfile(os.path.join(_qstack_qml_path, 'qstack_qml', '__init__.py')):
    import sys
    sys.path.insert(0,_qstack_qml_path)
    from qstack_qml import *
    sys.path.pop(0)
    del sys
else:
    try:
        from qstack_qml import *
    except ImportError:
        pass
del os, _qstack_qml_path
