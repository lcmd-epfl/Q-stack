# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import shutil
from pathlib import Path
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Q-stack'
copyright = '2023, LCMD'
author = 'LCMD'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'sphinx.ext.autodoc', 
        'sphinx.ext.napoleon', 
        'myst_parser',
        'sphinx.ext.todo'
        ]

todo_include_todos = False
todo_emit_warnings = False

note_include_todos = False

source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = []

suppress_warnings = [ 'ref.myst']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_theme_options = {
        "logo": "../images/logo.png",
        "logo_name": False,
        "description": "Codes for pre- and post-processing tasks for QML.",
        "logo_text_align": True,
        "github_button": True,
        "github_repo": "Q-stack",
        "github_user": "lcmd-epfl",
        }
#html_theme = 'classic'
#html_theme = 'bizstyle'
#html_theme = 'basicstrap'
#html_theme_options = {
#         'rightsidebar': False,
#         'content_fixed': True,
#         'content_width': '90%',
#         'header_inverse': False,
#         'relbar_inverse': False,
#        'noresponsive': False,
#        'inner_theme': True,
#        'inner_theme_name': 'bootswatch-cerulean',
        #'inner_theme_name': 'bootswatch-journal',
#        }

# basicstrap html theme options in:
# https://pythonhosted.org/sphinxjp.themes.basicstrap/design.html

html_static_path = ['_static']

# -- Images from README to HTML --------------------------------------------

def _copy_readme_images(app, exception):
    # Only run if build succeeded
    if exception:
        return
    # Source folder where you keep images during doc build
    src = Path(app.srcdir) / "images"         # docs/source/images
    # Destination that matches README's <img src="./images/..."> from api/index.html
    dst = Path(app.outdir) / "images" # _build/html/api/images

    if not src.exists():
        return

    for p in src.rglob("*"):
        if p.is_file():
            target = dst / p.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)

def setup(app):
    app.connect("build-finished", _copy_readme_images)

