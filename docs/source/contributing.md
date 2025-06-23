# Contribute to documentation

The documentation in the `jax-ai-stack` repository is meant to build on documentation
of individual packages, and specifically cover topics that touch on multiple packages
in the stack. If you see something missing and would like to contribute, please first
[open an issue](https://github.com/jax-ml/jax-ai-stack/issues/new) and let us know what
you have in mind!

## Pre-requisites

To contribute to the documentation, you will need to set your development environment.

You can create a virtual environment or conda environment and install the packages in `docs/requirements.txt` by running

```bash
pip install -r docs/requirements.txt
```

from the root of the repository.

## Documentation via Jupyter notebooks

The `jax-ai-stack` documentation includes Jupyter notebooks that are rendered
directly into the website via the [myst-nb](https://myst-nb.readthedocs.io/) extension.
To ease review and diff of notebooks, we keep markdown versions of the content
synced via [jupytext](https://jupytext.readthedocs.io/).

Note you will need to install `jupytext` to sync the notebooks with markdown files:

```bash
# With pip
python -m pip install jupytext

# With conda
conda install -c conda-forge jupytext
```

### Adding a new notebook

We aim to have one notebook per topic or tutorial covered.
To add a new notebook to the repository, first move the notebook into the appropriate
location in the `docs` directory:

```bash
mv ~/new-tutorial.ipynb docs/source/new_tutorial.ipynb
```

Next, we use `jupytext` to mark the notebook for syncing with Markdown:

```bash
jupytext --set-formats ipynb,md:myst docs/source/new_tutorial.ipynb
```

Finally, we can sync the notebook and markdown source:

```bash
jupytext --sync docs/source/new_tutorial.ipynb
```

To ensure that the new notebook is rendered as part of the site, be sure to add
references to a `toctree` declaration somewhere in the source tree, for example
in `docs/source/tutorials.md` or `docs/source/examples.md`.
You will also need to add references in `docs/conf.py`
to specify whether the notebook should be executed, and to specify which file
sphinx should use when generating the site.

### Editing an existing notebook

When editing the text of an existing notebook, it is recommended to edit the
markdown file only, and then automatically sync using `jupytext` via the
`pre-commit` framework, which we use to check in GitHub CI that notebooks are
properly synced.
For example, say you have edited `docs/new_tutorial.md`, then
you can do the following:

```bash
pip install pre-commit
git add docs/source/new_tutorial.*          # stage the new changes
pre-commit run                       # run pre-commit checks on added files
git add docs/source/new_tutorial.*          # stage the files updated by pre-commit
git commit -m "update new tutorial"  # commit to the branch
```

## Building the documentation locally

To build the documentation locally, you can run the following command:

```bash
# from the root of the repository
sphinx-build -b html docs/source docs/_build/html
```

You can then open the generated HTML files in your browser by opening `docs/_build/html/index.html`.
