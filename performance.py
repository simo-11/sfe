"""A simple performance test.

This script is used to generate the table in README.md.

"""
import timeit
import itertools
import numpy as np
import skfem as sf
import skfem.models.poisson as sfp
def pre(N=3):
    m = sf.MeshTet.init_tensor(*(3 * (np.linspace(0., 1., N),)))
    return m

print('| Degrees-of-freedom | Assembly (s) | Linear solve (s) |')
print('| --- | --- | --- |')

def assembler(m):
    basis = sf.Basis(m, sf.ElementTetP1())
    return (
        sfp.laplace.assemble(basis),
        sfp.unit_load.assemble(basis),
    )

for k in itertools.chain((6,),range(6, 14)):
    m = pre(int(2 ** (k / 3)))
    assemble_time = timeit.timeit(lambda: assembler(m), number=1)
    A, b = assembler(m)
    D = m.boundary_nodes()
    solve_time = timeit.timeit(lambda:
        sf.solve(*sf.condense(A, b, D=D)), number=1)
    print(f'| {len(b)} | {assemble_time:.5f} | {solve_time:.5f} |')