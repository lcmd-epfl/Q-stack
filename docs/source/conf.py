# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
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

todo_include_todos = True
todo_emit_warnings = False

note_include_todos = True

source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = []

suppress_warnings = [ 'ref.myst']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
#html_theme = 'classic'
#html_theme = 'bizstyle'
html_theme = 'basicstrap'
html_theme_options = {
         'rightsidebar': False,
         'content_fixed': True,
         'content_width': '90%',
         'header_inverse': False,
         'relbar_inverse': False,
        'noresponsive': False,
        'inner_theme': True,
        'inner_theme_name': 'bootswatch-cerulean',
        #'inner_theme_name': 'bootswatch-journal',
        }

# basicstrap html theme options in:
# https://pythonhosted.org/sphinxjp.themes.basicstrap/design.html

html_static_path = ['_static']
