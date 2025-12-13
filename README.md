[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
==============================================
Q-stack
==============================================
<p align="center"><img alt="qstack logo" align="center" src="./images/logo.png" width=66%></p>


## About

Q-stack is a stack of codes for dedicated pre- and post-processing tasks for Quantum Machine Learning (QML).


## Install

The installation of the library for python use can be done executing one of the following commands:

```bash
python -m pip install git+https://github.com/lcmd-epfl/Q-stack.git
python -m pip install "qstack @ git+https://github.com/lcmd-epfl/Q-stack.git"
python -m pip install "qstack[all] @ git+https://github.com/lcmd-epfl/Q-stack.git"
```

The last one is recommended if you do not know which features you wish to use, since they pull the dependencies required with all 'optional' parts of Q-stack.

### Features:

| feature | capability  |
|---|---|
| `[pyscf]` | quantum-chemical computations, wrappers, and conversion tools |
| `[qml]` | stucture-based representations such as SLATM, SLATM_d, B2R2 (no quantum chemistry) |
| `[spahm]` | SPAHM (ε-SPAHM, SPAHM(a,b)) representations |
| `[cell2mol]` | interface with [cell2mol](https://github.com/lcmd-epfl/cell2mol) |
| `[equio]`   | interface with [metatensor](https://github.com/metatensor/metatensor) |
| `[regression]` | tools to run kernel ridge regression (no quantum chemistry) | 
| `[wigner]` | tool to compute Wigner d-matrices for real spherical harmonics |
| `[dev]` | all above + linting + documentation building |


If you want to be able to edit Q-stack's code after installing, you need to download it yourself, for instance with this series of commands

```bash
git clone https://github.com/lcmd-epfl/Q-stack
cd Q-stack

#optionally, run the following line, if you want to be extra-careful about reproducibility, by installing a well-known version of all dependencies
python -m pip install -r requirements.py3.13.txt   # (or "requirements.py3.11.txt" if you have an older version of python and that first file doesn't work)

python -m pip install -e .[all]    # note: this translates as "install as '-e'ditable, install from this directory ('.'), with optional feature 'all'"
```

For the optional step above, we also have an `environment.yml` file available, if you prefer working with conda environments.


## Examples
Q-stack comes with several example codes that illustrate some of its key capabilities.
Check out the `examples/` and `tests/` folders:

* Field decomposition:
  - [`examples/example_deco.py`](https://github.com/lcmd-epfl/Q-stack/tree/master/examples/example_deco.py)
  - [`tests/test_fitting.py`](https://github.com/lcmd-epfl/Q-stack/tree/master/tests/test_fitting.py)

* Computation of Hirshfeld charges:
  - [`examples/example_hirsh.py`](https://github.com/lcmd-epfl/Q-stack/tree/master/examples/example_hirsh.py)

* Basis set optimization:
  - [`examples/example_opt.py`](https://github.com/lcmd-epfl/Q-stack/tree/master/examples/example_opt.py)

* Generation of the ε-SPAHM representation:
  - [`examples/example_SPAHM.py`](https://github.com/lcmd-epfl/Q-stack/tree/master/examples/example_SPAHM.py)
  - [`tests/test_spahm.py`](https://github.com/lcmd-epfl/Q-stack/tree/master/tests/test_spahm.py)
  - [`tests/test_spahm_grad.py`](https://github.com/lcmd-epfl/Q-stack/tree/master/tests/test_spahm_grad.py)

* Generation of the SPAHM(a,b) representations:
  - [`tests/test_spahm_a.py`](https://github.com/lcmd-epfl/Q-stack/tree/master/tests/test_spahm_a.py)
  - [`tests/test_spahm_b.py`](https://github.com/lcmd-epfl/Q-stack/tree/master/tests/test_spahm_b.py)
  - [`tests/test_spahm_b_selected.py`](https://github.com/lcmd-epfl/Q-stack/tree/master/tests/test_spahm_b_selected.py)

* Reaction representations ($B^2R^2$ and $\mathrm{SLATM}_d$):
  - [`tests/test_rxn-repr.py`](https://github.com/lcmd-epfl/Q-stack/tree/master/tests/test_rxn-repr.py)

* Computation of DORI (density overlap regions indicator):
  - [`tests/test_dori.py`](https://github.com/lcmd-epfl/Q-stack/tree/master/tests/test_dori.py)

* Interface with [cell2mol](https://github.com/lcmd-epfl/cell2mol):
  - [`tests/test_c2mio.py`](https://github.com/lcmd-epfl/Q-stack/tree/master/tests/test_c2mio.py)

* Parsing ORCA binary output:
  - [`examples/example_orcaio.py`](https://github.com/lcmd-epfl/tree/master/examples/example_orcaio.py)
  - [`tests/test_orca.py`](https://github.com/lcmd-epfl/tree/master/tests/test_orca.py)


## Sources

Relevant packages incorporated (among others):
* [https://github.com/lcmd-epfl/azo-xcite-tools](https://github.com/lcmd-epfl/azo-xcite-tools)
* [https://github.com/lcmd-epfl/SPAHM](https://github.com/lcmd-epfl/SPAHM)
* [https://github.com/lcmd-epfl/SPAHM-RHO](https://github.com/lcmd-epfl/SPAHM-RHO)
* [https://github.com/lcmd-epfl/RHO-Decomposition](https://github.com/lcmd-epfl/RHO-Decomposition)
* [https://github.com/lcmd-epfl/OTPD-basis](https://github.com/lcmd-epfl/OTPD-basis)
* [https://github.com/lcmd-epfl/b2r2-reaction-rep](https://github.com/lcmd-epfl/b2r2-reaction-rep)
* [https://github.com/lcmd-epfl/ml-density](https://github.com/lcmd-epfl/ml-density)


## References

* A. Fabrizio, A. Grisafi, B. Meyer, M. Ceriotti, and C. Corminboeuf,
  “Electron density learning of non-covalent systems”,
  Chem. Sci. **10**, 9492 (2019)
  [![DOI](https://img.shields.io/badge/DOI-10.1039%2FC9SC02696G-blue)](https://doi.org/10.1039/C9SC02696G)

* A. Fabrizio, K. R. Briling, D. D. Girardier, and C. Corminboeuf,
  “Learning on-top: regressing the on-top pair density for real-space visualization of electron correlation”,
  J. Chem. Phys. **153**, 204111 (2020)
  [![DOI](https://img.shields.io/badge/DOI-10.1063%2F5.0033326-blue)](https://doi.org/10.1063/5.0033326)

* S. Vela, A. Fabrizio, K. R. Briling, and C. Corminboeuf,
  “Machine-learning the transition density of the productive excited states of azo-dyes”
  J. Phys. Chem. Lett. **12**, 5957–5962 (2021)
  [![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs.jpclett.1c01425-blue)](https://doi.org/10.1021/acs.jpclett.1c01425)

* K. R. Briling, A. Fabrizio, and C. Corminboeuf,
  “Impact of quantum-chemical metrics on the machine learning prediction of electron density”,
  J. Chem. Phys. **155**, 024107 (2021)
  [![DOI](https://img.shields.io/badge/DOI-10.1063/5.0055393-blue)](https://doi.org/10.1063/5.0055393)

* A. Fabrizio, K. R. Briling, and C. Corminboeuf,
  “SPAHM: the Spectrum of Approximated Hamiltonian Matrices representations”,
  Digital Discovery **1**, 286–294 (2022)
  [![DOI](https://img.shields.io/badge/DOI-10.1039/D1DD00050K-blue)](https://doi.org/10.1039/D1DD00050K)

* K. R. Briling, Y. Calvino Alonso, A. Fabrizio, and C. Corminboeuf,
  “SPAHM(a,b): Encoding the Density Information from Guess Hamiltonian in Quantum Machine Learning Representations”,
  J. Chem. Theory Comput. **20**, 1108–1117 (2024)
  [![DOI](https://img.shields.io/badge/DOI-10.1021/acs.jctc.3c01040-blue)](https://doi.org/10.1021/acs.jctc.3c01040)

* P. van Gerwen, A. Fabrizio, M. Wodrich, and C. Corminboeuf,
  “Physics-based representations for machine learning properties of chemical reactions”,
  Mach. Learn.: Sci. Technol. **3**, 045005 (2022)
  [![DOI](https://img.shields.io/badge/DOI-10.1088/2632--2153/ac8f1a-blue)](https://doi.org/10.1088/2632-2153/ac8f1a)


## Acknowledgements
The authors of Q-stack acknowledge the National Centre of Competence in Research (NCCR)
"Materials' Revolution: Computational Design and Discovery of Novel Materials (MARVEL)" 
of the Swiss National Science Foundation (SNSF, grant number 182892)
and the European Research Council (ERC, grant agreement no 817977).

![acknowledgements logos](./images/ackw.png)
