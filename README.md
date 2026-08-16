# sfe
[scikit-fem](https://github.com/kinnala/scikit-fem) and other examples
 * scikit-fem is a library for performing finite element assembly which provides also for solving small systems
 * petsc - the Portable, Extensible Toolkit for Scientific Computation
 * [taichi](https://www.taichi-lang.org/) provides Productive, portable, and performant GPU programming in Python
   * taichi does not support latest python versions and taichi-forge does not work out of box, so studies postponed in this context
   * [taichi](https://github.com/taichi-dev/taichi)

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
C:\> uv python install 3.14.7
github\scikit-fem [main ≡]> uv venv --clear --python 3.14.7
Using CPython 3.14.7  
Creating virtual environment at: .venv
Activate with: .venv\Scripts\activate
github\scikit-fem [main ≡]> .venv\Scripts\activate
```

## [Spyder IDE](https://www.spyder-ide.org/)
Istalling using winget
```
PS C:\> winget install -e --id Spyder.Spyder
```
Tools/Preferences/Python interpreter github/sfe/.venv/Scripts/python.exe
Working directory (right upper corner) github/sfe
Run,  configuration per file, Custom configuration, Advanced settings: Run in console's namescape instead of and empty one
for all used ones at least Debugger and IPython

## Python packages

Dependencies can be installed using pip
 * spyder-kernels - needed for spyder integration
 * scikit-fem - main target for this repo, requires numpy and scipy [all] also brings also matplotlib
 * gmsh - geometry and mesh definitions
 * pyvistaqt pyqt5 - requires pyvista and qtpy
 * vedo visualization using vtk
 * tqdm - is a Python library that provides a fast, extensible progress bar for loops and other iterable objects
 * mpi4py - message passing library for petsc which does not work with windows
 * petsc4py slepc4py - python interfaces for petsc
 ** does not work 
 * pytest spyder-unittest - helps running scikit-fem tests with debugger, as noted in https://docs.spyder-ide.org/current/plugins/unittest.html this combination is not working currently
 * taichi-forge - taichi version which support current components https://pypi.org/project/taichi-forge/, does not work so taichi postponed

Typical command needed after update of python is (uv in front if it is used)
```
uv pip install spyder-kernels==3.1.* scikit-fem[all] pyvistaqt pyqt5 gmsh vedo tqdm
```

# Getting  examples

 * Get code, e.g. by cloning or forking sfe repository
```
PS C:\Users\simon> git clone https://github.com/simo-11/sfe
```
 * Start spyder. Spyder can be started also from menus and .py files can be opened from menus
```
PS C:\Users\simon\sfe> C:\ProgramData\spyder-6\envs\spyder-runtime\Scripts\spyder.exe warping.py
```

# Examples

## warping.py
Running file imports needed modules and defines a few functions that are described below
 * test_circle calculates section properties for unit circle using selected elements, meshes and count of refinements. Warping constant gamma should be close to zero and area near pi.
   * Best results are obtained using ElementTriP2 and MeshTri2 which provide good results using 2 refinements (41 dofs), area=3.13915 and gamma=1.41854E-34
   * ElementTriP2 and MeshTri2 (85 dofs) area=3.13915 and gamma=5.61194E-09. Implementing MeshTri3 could provide better results.
* test_elements uses gmsh and meshio for CIRCLE, RECTANGLE, U and RHS. U and RHS are modelled using rounded corners.
   * results can be compared with e.g. https://rakenteidenmekaniikka.journal.fi/article/view/163217 
* test_circle_areas points out that the mesh defines integration limits and element type does not affect the calculation of area.

## vtkQuadraticTriangle.py
Test of smoothing by tessellation. For pyvista.tessellation i.e.
method tessellate in module pyvista.core.filters.data_set to be effective element size should be >1. 
Parameters can be tuned by using vtk.vtkTessellatorFilter and effect of parameters can be seen using vtkQuadraticTriangle.py.
max_n_subdivide/MaximumNumberOfSubdivisions can be se also in pyvista interface and default value is 3.
Default for ChordError is 0.001 which is not suitable for dense meshes using SI(metric) units.
Suitable value is about 0.001 times smallest arc i.e. 1e-6 if roundings of about 1 mm (0.001 m) are used.
