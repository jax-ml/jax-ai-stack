# Developer docs

## Contributing to JAX AI Stack Documentation
The documentation in the `jax-ai-stack` repository is meant to build on documentation
of individual packages, and specifically cover topics that touch on multiple packages
in the stack. If you see something missing and would like to contribute, please first
[open an issue](https://github.com/jax-ml/jax-ai-stack/issues/new) and let us know what
you have in mind!

## Documentation via Jupyter notebooks
The jax-ai-stack documentation includes a number of Jupyter notebooks which are rendered
directly into the website via the [myst-nb](https://myst-nb.readthedocs.io/) extension.
In order to ease review and diff of notebooks, we keep markdown versions of the content
synced via [jupytext](https://jupytext.readthedocs.io/).

### Adding a new notebook
To add a new notebook to the repository, first move the notebook into the appropriate
location in the `docs` directory. For example, you could do something like this:
```
mv ~/new-tutorial.ipynb docs/new_tutorial.ipynb
```
Next, we use jupytext to mark the notebook for syncing with markdown:
```
pip install jupytext
jupytext --set-formats ipynb,md:myst docs/new_tutorial.ipynb
```
Finally, we can sync the notebook and markdown source:
```
jupytext --sync docs/new_tutorial.ipynb
```
To ensure that the new notebook is rendered as part of the site, be sure to add
references to a `toctree` declaration somewhere in the source tree, for example
in `docs/tutorials.md`. You will also need to add references in `docs/conf.py`
to specify whether the notebook should be executed, and to specify which file
sphinx should use when generating the site.

### Editing an existing notebook
When editing the text of an existing notebook, it is recommended to edit just
the markdown file, and then automatically sync using `jupytext` via the
`pre-commit` framework, which we use to check in github CI that notebooks are
properly synced. For example, say you have edited `docs/new_tutorial.md`, then
you can do the following:
```
pip install pre-commit
git add docs/new_tutorial.*          # stage the new changes
pre-commit run                       # run pre-commit checks on added files
git add docs/new_tutorial.*          # stage the files updated by pre-commit
git commit -m "update new tutorial"  # commit to the branch
```
