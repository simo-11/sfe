# sfe
[scikit-fem](https://github.com/kinnala/scikit-fem) examples, scikit-fem is a library for performing finite element assembly 
which provides also for solving small systems

# Installations

## [Winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/)

## GIT

```
PS C:\> winget install -e --id Git.Git
```

## UV
```
PS C:\> winget install -e --id astral-sh.uv
```

## Python
```
C:\> uv python install 3.14
github\sfe [main ≡]> uv venv --clear --python 3.14
Using CPython 3.14.3
Creating virtual environment at: .venv
Activate with: .venv\Scripts\activate
github\sfe [main ≡]> .venv\Scripts\activate
```

## [Spyder IDE](https://www.spyder-ide.org/)
Istalling using winget
```
PS C:\> winget install -e --id Spyder.Spyder
```
Tools/Preferences/Python interpreter github/sfe/.venv/Scripts/python.exe
Working directory (right upper corner) github/sfe
Run configuration per file, Custom configuration, Advanced settings: Run in console's namescape instead of and empty one

## Python packages

Dependencies can be installed using pip
 * spyder-kernels - needed for spyder integration
 * scikit-fem - main target for this repo, requires numpy and scipy [all] also brings also matplotlib
 * pyvistaqt pyqt5 - requires pyvista and qtpy

Typical command needed after update of python is (uv in front if it is used)
```
uv pip install spyder-kernels==3.1.* scikit-fem[all] pyvistaqt pyqt5
```

## Using sfe

 * Get code, e.g. by cloning or forking sfe repository
```
PS C:\Users\simon> git clone https://github.com/simo-11/sfe
```
 * Start spyder. Spyder can be started also from menus and main_cells.m can be opened from menus
```
PS C:\Users\simon\sfe> C:\ProgramData\spyder-6\envs\spyder-runtime\Scripts\spyder.exe main_cells.py
```
