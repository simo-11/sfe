# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 18:33:14 2026

@author: simon
"""
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
# %% Parameters
E = 1e9
nu = 0.3
G = E/(2*(1+nu))
g=9.8
A=0.0044
rho=1000
It = 10.513e-6
Iw = 1.64e-9
Ixx=1.14e-5
Iyy=4.18e-5
Ixy=1.41e-5
L = 2
M = 1000.0
plt_pause=0.5
res={}
# %% Differential equation for torsion for solve_bvp
# GI_t*theta'' - EIw*theta'''' = m
def torsion_fun(x, y):
    # y[0]=theta, y[1]=theta', y[2]=theta'', y[3]=theta'''
    dydx = np.vstack((y[1], y[2], y[3], (G*It/E/Iw)*y[2]))
    return dydx

def torsion_bc(ya, yb):
    # ya = value at x=0, yb = values at x=L
    return np.array([
        ya[0],        # theta(0)=0
        ya[1],        # theta'(0)=0
        yb[2],        # theta''(L)=0
        G*It*yb[1] - E*Iw*yb[3] - M  # moment
    ])
# %% bvp
x = np.linspace(0, L, 100)
y_init = np.zeros((4, x.size))
uc='bvp'
res[uc]=solve_bvp(torsion_fun, torsion_bc, x, y_init)
# %% plot
fig,axes=plt.subplot_mosaic(
[
    ["rotation"],
    ["moment"],
]
,num='torsion results',clear=True)
for uc,sol in res.items():
    ax=axes["rotation"]
    ax.plot(sol.x, 180/np.pi*sol.y[0], label=uc)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("θ(x) [deg]")
    ax.legend()
    plt.pause(plt_pause)
    T = G*It*sol.y[1] - E*Iw*sol.y[3]
    ax=axes["moment"]
    ax.plot(sol.x, T, label=f"Total moment for {uc}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Moment [Nm]")
    T = G*It*sol.y[1]
    ax.plot(sol.x, T, label=f"Moment from It for {uc}")
    T = - E*Iw*sol.y[3]
    ax.plot(sol.x, T, label=f"Moment from Iw for {uc}")
    ax.legend()
    plt.pause(plt_pause)
#
