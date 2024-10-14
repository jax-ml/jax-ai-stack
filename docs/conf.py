# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'jax-ai-stack'
copyright = '2024, JAX team'
author = 'JAX team'
release = '0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',
]

templates_path = ['_templates']
exclude_patterns = []
source_suffix = ['.rst', '.ipynb', '.md']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = []

exclude_patterns = [
    # Sometimes sphinx reads its own outputs as inputs!
    'build/html',
    'build/jupyter_execute',
    # Exclude markdown sources for notebooks:
    'digits_vae.md',
    'getting_started_with_jax_for_AI.md',
]

suppress_warnings = [
    'misc.highlighting_failure',  # Suppress warning in exception in digits_vae
]

# -- Options for myst ----------------------------------------------
myst_heading_anchors = 3  # auto-generate 3 levels of heading anchors
myst_enable_extensions = ['dollarmath']
nb_execution_mode = 'force'
nb_execution_allow_errors = False
nb_merge_streams = True
nb_execution_show_tb = True

# Notebook cell execution timeout; defaults to 30.
nb_execution_timeout = 100

# List of patterns, relative to source directory, that match notebook
# files that will not be executed.
nb_execution_excludepatterns = []
