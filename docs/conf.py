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
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = []
source_suffix = ['.rst', '.ipynb', '.md']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_title = 'JAX AI Stack'
html_static_path = ['_static']

# Theme-specific options
# https://sphinx-book-theme.readthedocs.io/en/stable/reference.html
html_theme_options = {
    'show_navbar_depth': 2,
    'show_toc_level': 2,
    'repository_url': 'https://github.com/jax-ml/jax-ai-stack',
    'path_to_docs': 'docs/',
    'use_repository_button': True,
    'navigation_with_keys': True,
}

exclude_patterns = [
    # Sometimes sphinx reads its own outputs as inputs!
    'build/html',
    'build/jupyter_execute',
    # Exclude markdown sources for notebooks:
    'digits_vae.md',
    'getting_started_with_jax_for_AI.md',
    'JAX_for_PyTorch_users.md',
    'JAX_porting_PyTorch_model.md',
    'digits_diffusion_model.md',
    'JAX_for_LLM_pretraining.md',
    'JAX_basic_text_classification.md',
    'JAX_examples_image_segmentation.md',
    'JAX_Vision_transformer.md',
    'JAX_machine_translation.md',
]

suppress_warnings = [
    'misc.highlighting_failure',  # Suppress warning in exception in digits_vae
]

# -- Options for myst ----------------------------------------------
myst_heading_anchors = 3  # auto-generate 3 levels of heading anchors
myst_enable_extensions = [
    'dollarmath',
    'linkify',
]
nb_execution_mode = 'force'
nb_execution_allow_errors = False
nb_merge_streams = True
nb_execution_show_tb = True

# Notebook cell execution timeout; defaults to 30.
nb_execution_timeout = 100

# List of patterns, relative to source directory, that match notebook
# files that will not be executed.
nb_execution_excludepatterns = [
    'JAX_for_PyTorch_users.ipynb',
    'JAX_porting_PyTorch_model.ipynb',
    'digits_diffusion_model.ipynb',
    'JAX_for_LLM_pretraining.ipynb',
    'JAX_basic_text_classification.ipynb',
    'JAX_examples_image_segmentation.ipynb',
    'JAX_Vision_transformer.ipynb',
    'JAX_machine_translation.ipynb',
]
