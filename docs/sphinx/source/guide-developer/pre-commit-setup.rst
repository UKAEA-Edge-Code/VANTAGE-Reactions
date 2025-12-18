**************************
Setup for pre-commit hooks
**************************

Pre-requisites
==============

* python 3.8+: Tested up to python 3.12
* (Optional) pipx : Tested for 1.4.3

Installation of pre-commit
==========================
There are a few different ways to install ``pre-commit``. The easiest, if available, is to use ``pipx``
::

    pipx install pre-commit

Alternatively, you can create a virtual environment with ``venv`` and ``pip install`` inside it or use ``uv`` or ``pixi`` or even Spack (listed as ``py-pre-commit`` in Spack).

Generate the hook scripts
=========================
Assuming that VANTAGE-Reactions has been cloned per the instructions in ``User Guide/Installation``, then inside the repo directory, run:
::

    pre-commit install

How it works
============
Now every time you make a commit, any C++ files that have been changed will go through the process of being checked by ``clang-format``. If it finds incorrect formatting, it will make the necessary changes and you will have to re-stage the files that it has modified before completing the commit (at which point the ``clang-format`` test will pass and let you commit).
If you want to commit without going through the ``clang-format`` step, then it is possible to use the ``--no-verify`` option when committing. In any case, your branch will be subject to a ``clang-format`` check upon opening a pull request as well but it is good practice to have this enabled locally to make merging a bit easier (otherwise another commit with the required changes will be necessary).
