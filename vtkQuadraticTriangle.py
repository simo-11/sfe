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

tri = vtk.vtkQuadraticTriangle()
for i in range(6): tri.GetPointIds().SetId(i, i)

# 3. Grid ja Mapper
grid = vtk.vtkUnstructuredGrid()
grid.SetPoints(pts)
grid.InsertNextCell(tri.GetCellType(), tri.GetPointIds())
pv_mesh = pv.wrap(grid)
smooth_mesh = pv_mesh.tessellate()

if not "mpv" in globals():
    mpv = pyvistaqt.MultiPlotter(nrows=1, ncols=2)
    mpv._window.setWindowTitle("VTK mesh tesselation")
amp=mpv[0,0]
amp.clear()
amp.add_text('Smoothed with tessellate'
             ,position='upper_edge'
             ,font_size=12)
amp.add_mesh(smooth_mesh
             ,style="wireframe"
             ,color="blue"
             ,line_width=2)
amp.add_mesh(smooth_mesh, opacity=0.3, color="cyan")
amp.add_points(pv_mesh.points, color="red", point_size=10)
amp=mpv[0,1]
amp.clear()
amp.add_mesh(pv_mesh
             ,style="wireframe"
             ,color="blue"
             ,line_width=2)
amp.add_mesh(pv_mesh, opacity=0.3, color="cyan")
amp.add_points(pv_mesh.points, color="red", point_size=10)
mpv.show()

"""
actor = vtk.vtkActor()
actor.SetMapper(mapper)
ren = vtk.vtkRenderer()
ren.AddActor(actor)
win = vtk.vtkRenderWindow()
win.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(win)
win.Render()
iren.Start()
"""