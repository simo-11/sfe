# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:49:13 2026

@author: simon
"""
import numpy as np
import vtk

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
def apply_gravity():
    """Apply gravity to velocities."""
    global v
    v += gravity * dt
    v[inv_mass == 0.0] = 0.0


def integrate():
    """Explicit Euler integration."""
    global x, v
    x += v * dt
    v *= damping


def solve_springs():
    """XPBD-like spring solve with elastic + plastic strain and fracture."""
    global x, rest_length, plastic_strain, active

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


# -------------------------
# VTK visualization setup
# -------------------------
points = vtk.vtkPoints()
for i in range(N):
    points.InsertNextPoint(x[i, 0], x[i, 1], 0.0)

lines = vtk.vtkCellArray()
for i in range(N - 1):
    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, i)
    line.GetPointIds().SetId(1, i + 1)
    lines.InsertNextCell(line)

polydata = vtk.vtkPolyData()
polydata.SetPoints(points)
polydata.SetLines(lines)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(polydata)

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(1.0, 1.0, 0.2)
actor.GetProperty().SetLineWidth(3.0)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.1, 0.2)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 600)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# initial downward velocity at right end (simulates load)
v[-1, 1] = -2.0


def timer_callback(obj, event):
    """VTK timer callback: advance simulation and update geometry."""
    global x, v

    for _ in range(substeps):
        apply_gravity()
        integrate()
        solve_springs()

    # update VTK points
    for i in range(N):
        points.SetPoint(i, float(x[i, 0]), float(x[i, 1]), 0.0)
    points.Modified()
    polydata.Modified()
    render_window.Render()


interactor.Initialize()
render_window.Render()

timer_id = interactor.CreateRepeatingTimer(10)  # ms
interactor.AddObserver("TimerEvent", timer_callback)

interactor.Start()

