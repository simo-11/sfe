# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:49:13 2026

@author: simon
"""
import numpy as np
import pyvista as pv
from warping import start_mp

def xpbd_beam():
    # -------------------------
    # Parameters
    # -------------------------
    N = 20                 # number of particles
    length = 1.0
    dt = 1.0 / 240.0
    substeps = 10
    gravity = np.array([0.0, -9.81], dtype=np.float64)

    mass_total = 1.0
    mass = mass_total / N
    inv_mass = np.ones(N) / mass
    inv_mass[0] = 0.0      # left end fixed

    k = 5e4                # spring stiffness
    yield_strain = 0.01    # elastic limit (yield point)
    plastic_rate = 0.2     # how fast plastic strain accumulates
    fracture_strain = 0.3  # fracture threshold (set high to disable)

    damping = 0.999

    # -------------------------
    # State arrays
    # -------------------------
    x = np.zeros((N, 2), dtype=np.float64)
    v = np.zeros((N, 2), dtype=np.float64)

    # initialize bar horizontally
    for i in range(N):
        t = i / (N - 1)
        x[i] = np.array([t * length, 0.0])

    # springs between neighbors
    num_springs = N - 1
    spring_i = np.arange(0, N - 1, dtype=np.int32)
    spring_j = np.arange(1, N, dtype=np.int32)

    rest_length = np.linalg.norm(x[spring_j] - x[spring_i], axis=1)
    plastic_strain = np.zeros(num_springs, dtype=np.float64)
    active = np.ones(num_springs, dtype=bool)  # broken springs = False


    # -------------------------
    # Physics
    # -------------------------
    def apply_gravity(v):
        """Apply gravity to velocities."""
        v += gravity * dt
        v[inv_mass == 0.0] = 0.0


    def integrate(x,v):
        """Explicit Euler integration."""
        x += v * dt
        v *= damping


    def solve_springs():
        """XPBD-like spring solve with elastic + plastic strain and fracture."""

        for s in range(num_springs):
            if not active[s]:
                continue

            i = spring_i[s]
            j = spring_j[s]

            xi = x[i]
            xj = x[j]
            wi = inv_mass[i]
            wj = inv_mass[j]

            d = xj - xi
            L = np.linalg.norm(d)
            if L < 1e-8:
                continue
            n = d / L

            L0 = rest_length[s]
            strain = (L - L0) / L0

            # --- Plasticity: if strain exceeds yield, accumulate plastic strain ---
            if abs(strain) > yield_strain:
                delta_plastic = (abs(strain) - yield_strain) * np.sign(strain)
                plastic_strain[s] += plastic_rate * delta_plastic

                # update rest length according to accumulated plastic strain
                rest_length[s] = L / (1.0 + plastic_strain[s])

                # recompute strain with updated rest length
                L0 = rest_length[s]
                strain = (L - L0) / L0

            # --- Fracture: deactivate spring if strain too large ---
            if abs(strain) > fracture_strain:
                active[s] = False
                continue

            # --- Elastic correction (simplified XPBD) ---
            C = L - L0
            w_sum = wi + wj
            if w_sum == 0.0:
                continue

            lambda_ = -k * C / w_sum * dt * dt
            corr = lambda_ * n

            if wi > 0.0:
                x[i] += corr * wi
            if wj > 0.0:
                x[j] -= corr * wj



    # ----------------------------------------
    # Build polyline as PyVista PolyData
    # ----------------------------------------
    # x: (N, 2) array of node positions

    points = np.column_stack([x[:, 0], x[:, 1], np.zeros(N)])

    # Create connectivity: 0-1, 1-2, ..., N-2 - N-1
    lines = np.hstack([
        np.array([2, i, i + 1], dtype=np.int64)
        for i in range(N - 1)
    ])

    poly = pv.PolyData()
    poly.points = points
    poly.lines = lines

    # ----------------------------------------
    # PyVistaQt plotter
    # ----------------------------------------
    start_mp()
    mp_global = globals().get("mp")
    plotter=mp_global[0,0]
    plotter.add_mesh(
        poly,
        color=(1.0, 1.0, 0.2),
        line_width=3.0,
        render=False,
    )
    plotter.set_background(color=(0.1, 0.1, 0.2))

    # initial downward velocity at right end
    v[-1, 1] = -2.0

    # ----------------------------------------
    # Timer callback using Qt
    # ----------------------------------------
    def timer_callback():
        """Advance simulation and update geometry."""
        for _ in range(substeps):
            apply_gravity(v)
            integrate(x,v)
            solve_springs()

        # update point coordinates
        poly.points[:, 0] = x[:, 0]
        poly.points[:, 1] = x[:, 1]
        poly.points[:, 2] = 0.0

        poly.modified()
        plotter.render()
    if hasattr(plotter, "_cb_id"):
        plotter.remove_callback(plotter._cb_id)
        del plotter._cb_id
    cb_id=plotter.add_callback(timer_callback,interval=10)
    plotter._cb_id = cb_id

"""
xpbd_beam()
"""