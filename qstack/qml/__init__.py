try:
    import ase
except ImportError:
    print("""

ERROR: cannot import ase. Have you installed qstack with the \"qml\" option?\n\n
(for instance, with `pip install qstack[qml]` or `pip install qstack[all]`)

""")
    raise
else:
    from . import b2r2
    from . import slatm
    del ase

