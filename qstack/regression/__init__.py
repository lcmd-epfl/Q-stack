try:
    import sklearn
    del sklearn
except ImportError:
    print("""

ERROR: cannot import scikit-learn. Have you installed qstack with the \"regression\" option?\n\n
(for instance, with `pip install qstack[regression] or `pip install qstack[all]``)

""")
    raise
