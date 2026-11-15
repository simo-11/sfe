"""A simple performance test.

This script is adapted from sckit-fem where it
is used to generate the table in README.md.

"""
import time
import cc
import itertools
import numpy as np
import skfem as sf
import skfem.models.poisson as sfp
def pre(N=3):
    m = sf.MeshTet.init_tensor(*(3 * (np.linspace(0., 1., N),)))
    return m

print('| dofs | Assembly (s) | Condense (s) |'
       ' sf.solve (s) | sf residual |'
       ' amg.solve (s) | amg residual |'
       )
print('| --- | --- | --- | --- | --- | --- | --- |')

def assembler(m):
    basis = sf.Basis(m, sf.ElementTetP1())
    return (
        sfp.laplace.assemble(basis),
        sfp.unit_load.assemble(basis),
    )

do_amg_solve=False
ma=[]
for k in itertools.chain((6,),range(6, 14)):
    t=[]
    m = pre(int(2 ** (k / 3)))
    t.append(time.perf_counter())
    A, b = assembler(m)
    D = m.boundary_nodes()
    t.append(time.perf_counter())
    AII, bI, xD, I = sf.condense(A, b, D=D)
    t.append(time.perf_counter())
    ua=[]
    r=[]#residuals
    u=sf.solve(AII, bI, xD, I)
    t.append(time.perf_counter())
    ua.append(u)
    u,m=cc.amg_solve(AII, bI, xD, I,verb=False)
    ma.append(m)
    t.append(time.perf_counter())
    ua.append(u)
    for u in ua:
        r.append(np.linalg.norm(b-A@u)/np.linalg.norm(b))
    print((f'| {len(b)} | {t[1]-t[0]:.5f} | {t[2]-t[1]:.5f} |'
          f' {t[3]-t[2]:.5f} | {r[0]:1.3G} |'
          f' {t[4]-t[3]:.5f} | {r[1]:1.3G} |'
          ))