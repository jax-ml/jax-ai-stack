# Releasing jax-ai-stack

To create a new `jax-ai-stack` release, take the following steps:

1. Send a pull request updating the version in `pyproject.toml` and
   `jax-ai-stack/__init__.py` to the version based on the release date.
   Also update `CHANGELOG.md` with the pinned package versions.
2. Once this is merged, create the release tag and push it to github. An
   example from the 2024.10.1 release:
   ```
   $ git checkout main
   $ git pull upstream main  # upstream is github.com:jax-ml/jax-ai-stack.git
   $ git log  # view commit log & ensure the most recent commit
              # is your version update PR
   $ git tag -a v2024.10.1 -m "v2024.10.1 Release"
   $ git push upstream v2024.10.1
   ```
3. Navigate to https://github.com/jax-ml/jax-ai-stack/releases/new, and select
   this new tag. Copy the change description from `CHANGELOG.md` into the
   release notes, and click *Publish release*.
4. Publishing the release will trigger the CI jobs configured in
   `.github/workflows/wheels.yml`, which will build the wheels and source
   distributions and publish them to PyPI. Navigate to
   https://github.com/jax-ml/jax-ai-stack/actions/workflows/release.yml and
   look for the job associated with this release; monitor it to ensure it
   finishes green (this should take less than a minute).
5. Once the build is complete, check https://pypi.org/project/jax-ai-stack/
   to ensure that the new release is present.
