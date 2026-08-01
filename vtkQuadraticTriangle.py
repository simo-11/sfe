# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 12:29:19 2026
With help of Google Search AI
@author: simon
"""

import vtk
import pyvistaqt
import pyvista as pv
from PyQt5.QtWidgets import (QCheckBox, QWidget, QVBoxLayout, QSlider,
                             QDoubleSpinBox, QLabel, QDockWidget)
from PyQt5.QtCore import Qt
import numpy as np
import datetime
import math
import sys

scaler=1
chord_error=0.001
tess_subdivisions=4
add_wireframe=True

class CheckPanel(QWidget):
    """Checkbox panel calling build_meshes and updates global add_wireframe.
    """
    def __init__(self):
        super().__init__()
        self.cb = QCheckBox("Add wireframe")
        self.cb.setChecked(add_wireframe)
        self.cb.stateChanged.connect(self.on_change)
        layout = QVBoxLayout(self)
        layout.addWidget(self.cb)

    def on_change(self, state):
        global add_wireframe
        add_wireframe=state
        build_meshes()

class LogSlider(QWidget):
    """Logarithmic slider with redraw callback."""
    def __init__(self,var_name,min_value=0.001):
        super().__init__()
        self.var_name=var_name
        self.min_value=min_value
        self.log10_value=math.log10(min_value)
        # Slider controlling log10 value
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        logv = np.log10(globals()[var_name])
        v = np.interp(logv, [self.log10_value, 0], [0, 1000])
        self.slider.setValue(int(v))
        # Spinbox showing actual value
        self.spin = QDoubleSpinBox()
        self.spin.setRange(min_value, 1.0)
        decimals=int(abs(self.log10_value))
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(10*min_value)
        self.spin.setValue(globals()[var_name])
        fmt=f".{decimals}f"
        self.label = QLabel(f"{var_name} ({min_value:{fmt}}–1.0)")
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.spin)
        self.slider.sliderReleased.connect(self._on_release)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _from_slider(self, v):
        """Update spinbox from slider."""
        log_val = np.interp(v, [0, 1000], [self.log10_value, 0])
        x = 10 ** log_val
        self.spin.blockSignals(True)
        self.spin.setValue(x)
        self.spin.blockSignals(False)
        globals()[self.var_name]=x

    def _on_release(self):
        build_meshes()

    def _from_spin(self, x):
        """Update slider from spinbox."""
        mod = sys.modules[__name__]
        setattr(mod, self.var_name,x)
        log_val = np.log10(x)
        v = np.interp(log_val, [self.log10_value, 0], [0, 1000])
        self.slider.blockSignals(True)
        self.slider.setValue(int(v))
        self.slider.blockSignals(False)
        build_meshes()

def build_meshes():
    global mpv,dock,scaler,add_wireframe
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
    tess = vtk.vtkTessellatorFilter()
    tess.SetInputData(pv_mesh)
    tess.SetChordError(chord_error)
    tess.SetMaximumNumberOfSubdivisions(tess_subdivisions)
    tess.Update()
    smooth_mesh = pv.wrap(tess.GetOutput())

    if not "mpv" in globals():
        mpv = pyvistaqt.MultiPlotter(nrows=1, ncols=2)
        mpv._window.setWindowTitle("VTK mesh tessellation")
        mpv.show()
    amp=mpv[0,0]
    amp.clear()
    amp.add_text('Smoothed with tessellate'
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
    now=datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    amp.app_window.statusBar().showMessage(
    f"""scaler={scaler:.3f}, chord_error={chord_error:.5G}
   tess_subdivisions={tess_subdivisions}
 {now}""")
    if not "dock" in globals():
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(LogSlider('scaler'))
        layout.addWidget(LogSlider('chord_error',0.001*scaler))
        subdiv_label=QLabel(f'max subdivisions {tess_subdivisions}');
        layout.addWidget(subdiv_label)
        def update_tess_subdivisions(v):
            global tess_subdivisions
            tess_subdivisions=v
            subdiv_label.setText(f'max subdivisions {v}')
            build_meshes()
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(6)
        slider.setValue(tess_subdivisions)
        slider.valueChanged.connect(update_tess_subdivisions)
        layout.addWidget(slider)
        layout.addWidget(CheckPanel())
        layout.addStretch(1)
        aw=mpv[0,0].app_window
        dock = QDockWidget("Options", aw)
        dock.setFeatures(QDockWidget.DockWidgetClosable)
        dock.setMinimumSize(dock.minimumSizeHint())
        def on_dock_visibility(visible):
            global dock
            if not visible:
                if "dock" in globals():
                    del(dock)
        dock.visibilityChanged.connect(on_dock_visibility)
        dock.setWidget(container)
        aw.addDockWidget(Qt.RightDockWidgetArea, dock)

#%% run
if "dock" in globals():
    dock.close()
build_meshes()