# Instructions to create the documentation in html format

Install Sphinx:

```bash
pip install sphinx myst-parser sphinx-autobuild sphinx-ext-todo alabaster
```

on the main directory `Q-stack/`, execute the code `generate_rst.py` to update or generate the rst files used to build
the documentation:

```bash
python docs/generate_rst.py qstack/ -o docs/source/ --project "Qstack" --package-root-name qstack
```

Finally, in the directory `Q-stack/docs/` execute the comands:

```bash
make clean
make html
```

You can access the documenation on `Q-stack/docs/build/html/index.html`

