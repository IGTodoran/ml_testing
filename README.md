# MLOps repository

This repository gathers the MLOps initiatives assuming a pure Python implementation,  meaning that the
repository does not include any extensions that need to be compiled (ex: C++).

## Local Development

You will need to perform additional steps to enable convenient local development
of Python projects, namely steps required to extend the Python path.

### Editable Installation

If you want to work in virtual environments, you can perform an "editable
installation" to add your project to the Python path.

```
cd <repository_root>
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -e .[develop]
```

Here, `.` indicates the root of our project, and `develop` is the name of the
additional requirements group that we introduce in
[`pyproject.toml`](./pyproject.toml).

You can now run scripts such as `python3 scripts/sample_usage.py`, and the scripts
will correctly discover the content in `mlops` without you having to
explicitly change the Python path yourself.

**Caution:**

If you instrument this approach, you will need to add `*.egg-info` and `build`
to your [`.gitignore`](.gitignore) file. Also, if you change a module name or
location, you may need to re-run `pip install -e .` to fix import paths.
Finally, you will need to routinely make sure you are working with the latest
project dependencies in your local installation, particularly after you have
fetched latest changes from upstream.

### Docker

If you choose to develop inside a Docker container, in your Dockerfile, you can
set

```dockerfile
WORKDIR /path_to_your_bind_mounted_project
ENV PYTHONPATH="/path_to_your_bind_mounted_project:$PYTHONPATH"
```

This adds your bind-mounted project directory to the Python path.

Implementations of containerized development environments are not shown in this
reference project.

## File Structure

This project uses the following general structure.

```
.
├── .gitlab-ci       # CI workflow definitions.
├── docs             # Buildable prose and API-reference documentation.
├── mlops_lib        # Library-like code meant for reuse.
│   └── lib_sample
├── scripts          # Command-line entrypoints.
└── tests
    ├── integration  # Integration tests combining multiple project features.
    └── unit         # Unit tests for individual functions or classes.
```

## Testing

Unit test files are named after the fully-qualified module they provide tests
for. Individual tests are named after the fully-qualified module and then
function or class they cover. For example, see
[`tests/unit/test_mlops_lib_sample_sample.py`](./tests/unit/test_mlops_lib_sample_sample.py).

Integration tests are named in a way that describes the combination of features
they cover.

In this example, running all tests at once is done with just `pytest tests`.

## Linting

In this example, project-level linting is done by
[pre-commit](https://pre-commit.com/). Linting includes automatic code
formatters and static analyzers. For pre-commit, tool configuration is done in
[`.pre-commit-config.yaml`](.pre-commit-config.yaml).

Developers must run `pre-commit install` in the root of the repository to
install the git hooks. After this, pre-commit will automatically update
installed tools based on the contents of its configuration file, and it will
automatically run linting on new local commits (or during other stages, as
defined in the tooling configuration).

Recommended linters are at least:

- default pre-commit hooks: these are provided by `pre-commit sample-config`
- [black](https://github.com/psf/black): automatic code formatting
- [isort](https://github.com/PyCQA/isort): automatic organization of `import`s
- [mypy](https://github.com/python/mypy): static type checking based on type
  annotations in source code
- [pylint](https://github.com/pylint-dev/pylint): static analysis that catches
  common Python programming mistakes

**Caution:**

If you choose to use a development installation for local development, you may
find that some pre-commit hooks fail to import code in your source code modules.
If the hook that is failing is provided by pre-commit, it will run in a special,
isolated environment set up by pre-commit, and this environment may not be aware
of your local development installation of your project. To solve this, make the
underlying hook tool an explicit dependency in your project and configure the
hook to run locally.

For examples of this, see the explicit installation of `pylint` in
[pyproject.toml](./pyproject.toml) and the local hook configuration in
[.pre-commit-config.yaml](./.pre-commit-config.yaml).

## Typing

Python is a dynamically-typed language, but type annotations can help catch bugs
before even running your programs and are particularly helpful in communicating
interface semantics and requirements to other developers. Linters such as `mypy`
rely on type annotations to do their job. Type annotations are also picked up by
many documentation generators.

Add type annotations to code whenever possible. For an example, see the code in
[`mlops.lib_sample.sample`](./mlops/lib_sample/sample.py). When
supporting Python 3.8, you need to import some built-in types such as `List`
from `typing`. When using Python 3.9 or newer, the built-in types (ex: `list`)
can be used as subscriptable type annotations directly.

## Documentation

Note that all modules provide module-level documentation describing what the
module is for. Functions also provide documentation describing what they do, the
semantics and typing of the parameters, and the interface contracts (such as
parameter preconditions). We encourage this in all code.

For a concrete example, see
[`mlops.lib_sample.sample`](./mlops/lib_sample/sample.py).

Documentation is written in reStructuredText tailored for
[Sphinx](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).

Documentation should be automatically built by a GitLab CI/CD workflow.

## Continuous Integration (CI)

The test suite is run automatically against each merge request that wants to make
changes to the master branch. This should be done with GitLab CI/CD workflow.
This workflow additionally runs the linter on the entire
codebase to make sure the pull request introduces no violations.

## Pull Request Configuration

It is generally a good idea to require an approval from code owners before code
can be merged to your default branch. This should be enabled by defining
[GitLab Code Owners](https://docs.gitlab.com/ee/user/project/codeowners/).
