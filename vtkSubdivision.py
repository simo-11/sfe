# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:35:55 2026

@author: simon
"""
from vtkmodules.vtkFiltersModeling import vtkAdaptiveSubdivisionFilter
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import (
    vtkActor, vtkPolyDataMapper,
    vtkRenderWindow, vtkRenderer, vtkRenderWindowInteractor
)

# Lähde: nonlinear sphere (voi korvata omalla quadratic meshillä)
src = vtkSphereSource()
src.SetPhiResolution(5)
src.SetThetaResolution(5)
src.Update()

# Adaptive subdivision: tekee pinnasta kaarevan
adapt = vtkAdaptiveSubdivisionFilter()
adapt.SetInputConnection(src.GetOutputPort())
adapt.SetMaximumEdgeLength(0.01)   # säädä tarkkuutta
adapt.Update()

# Mapper
mapper = vtkPolyDataMapper()
mapper.SetInputConnection(adapt.GetOutputPort())

# Actor
actor = vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(0.2, 0.7, 1.0)
actor.GetProperty().EdgeVisibilityOn()
actor.GetProperty().SetEdgeColor(0, 0, 0)

# Renderer
ren = vtkRenderer()
ren.AddActor(actor)

# Window
renWin = vtkRenderWindow()
renWin.AddRenderer(ren)

# Interactor
iren = vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

renWin.Render()
iren.Start()

