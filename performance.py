"""A simple performance test.

This script is adapted from sckit-fem where it
is used to generate the table in README.md.

"""
import time
import types
import cc
import itertools
import numpy as np
import skfem as sf
def pre(N=3):
    m = sf.MeshTet.init_tensor(*(3 * (np.linspace(0., 1., N),)))
    return m
def assembler(m):
    basis = sf.Basis(m, sf.ElementTetP1())
    return (
        sf.models.poisson.laplace.assemble(basis),
        sf.models.poisson.unit_load.assemble(basis),
    )
print('| dofs | Assembly (s) | Condense (s) |'
       ' sf.solve (s) | sf residual |'
       ' amg.solve (s) | amg residual |'
       )
print('| --- | --- | --- | --- | --- | --- | --- |')

do_amg_solve=False
ma=[]
for k in itertools.chain((10,),range(10, 23, 3)):
    ns=types.SimpleNamespace()
    ns.t=[]
    ns.m = pre(k)
    ns.t.append(time.perf_counter())
    ns.A, ns.b = assembler(ns.m)
    ns.D = ns.m.boundary_nodes()
    ns.t.append(time.perf_counter())
    ns.AII, ns.bI, ns.xD, ns.I = sf.condense(ns.A, ns.b, D=ns.D)
    ns.t.append(time.perf_counter())
    ns.ua=[]
    ns.r=[]#residuals
    u=sf.solve(ns.AII, ns.bI, ns.xD, ns.I)
    ns.t.append(time.perf_counter())
    ns.ua.append(u)
    u,ns.ml=cc.amg_solve(ns.AII, ns.bI, ns.xD, ns.I,
                     return_solver=True,
                     verb=False)
    ma.append(ns)
    ns.t.append(time.perf_counter())
    ns.ua.append(u)
    for u in ns.ua:
        r = np.linalg.norm(ns.bI - ns.AII @ u[ns.I]) / np.linalg.norm(ns.bI)
        ns.r.append(r)
    print((f'| {len(ns.b)} | {ns.t[1]-ns.t[0]:.5f} | {ns.t[2]-ns.t[1]:.5f} |'
          f' {ns.t[3]-ns.t[2]:.5f} | {ns.r[0]:1.3G} |'
          f' {ns.t[4]-ns.t[3]:.5f} | {ns.r[1]:1.3G} |'
          ))