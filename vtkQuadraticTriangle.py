# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 12:29:19 2026
With help of Google Search AI
@author: simon
"""

import vtk
import pyvistaqt
import pyvista as pv

pts = vtk.vtkPoints()
pts.InsertNextPoint(0, 0, 0)
pts.InsertNextPoint(2, 0, 0)
pts.InsertNextPoint(1, 2, 0)
pts.InsertNextPoint(1, -0.5, 0)
pts.InsertNextPoint(1.8, 1, 0)
pts.InsertNextPoint(0.2, 1, 0)

poly = vtk.vtkPolyData()
poly.SetPoints(pts)
tr = vtk.vtkTransform()
scaler=0.01
tr.Scale(scaler, scaler, scaler)
tf = vtk.vtkTransformPolyDataFilter()
tf.SetTransform(tr)
tf.SetInputData(poly)
tf.Update()
tpts = tf.GetOutput().GetPoints()

tri = vtk.vtkQuadraticTriangle()
for i in range(6): tri.GetPointIds().SetId(i, i)

# 3. Grid ja Mapper
grid = vtk.vtkUnstructuredGrid()
grid.SetPoints(tpts)
grid.InsertNextCell(tri.GetCellType(), tri.GetPointIds())
pv_mesh = pv.wrap(grid)
smooth_mesh = pv_mesh.tessellate()

if not "mpv" in globals():
    mpv = pyvistaqt.MultiPlotter(nrows=1, ncols=2)
    mpv._window.setWindowTitle("VTK mesh tesselation")
amp=mpv[0,0]
amp.clear()
add_wireframe=False
amp.add_text(f'Smoothed with tessellate, scaler={scaler}'
             ,position='upper_edge'
             ,font_size=12)
if add_wireframe:
    amp.add_mesh(smooth_mesh
             ,style="wireframe"
             ,color="blue"
             ,line_width=2)
amp.add_mesh(smooth_mesh, opacity=0.3, color="cyan")
amp.add_points(pv_mesh.points, color="red", point_size=10)
amp=mpv[0,1]
amp.clear()
if add_wireframe:
    amp.add_mesh(pv_mesh
             ,style="wireframe"
             ,color="blue"
             ,line_width=2)
amp.add_mesh(pv_mesh, opacity=0.3, color="cyan")
amp.add_points(pv_mesh.points, color="red", point_size=10)
mpv.show()