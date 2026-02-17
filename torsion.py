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
M = 1000.0
plt_pause=0.1
res={}
# Differential equation for torsion for solve_bvp
# GI_t*theta'' - EIw*theta'''' = m
def bvp_fun(x, y):
    # y[0]=theta, y[1]=theta', y[2]=theta'', y[3]=theta'''
    dydx = np.vstack((y[1], y[2], y[3], (G*It/E/Iw)*y[2]))
    return dydx

def bvp_bc(ya, yb):
    # ya = value at x=0, yb = values at x=L
    return np.array([
        ya[0],        # theta(0)=0
        ya[1],        # theta'(0)=0
        yb[2],        # theta''(L)=0
        G*It*yb[1] - E*Iw*yb[3] - M  # moment
    ])
def bvp(L):
    x = np.linspace(0, L, 100)
    y_init = np.zeros((4, x.size))
    uc=f'bvp@{L}'
    res[uc]=solve_bvp(bvp_fun, bvp_bc, x, y_init)
def plot(L):
    fig,axes=plt.subplot_mosaic(
    [
        ["rotation"],
        ["moment"],
    ]
    ,num=f'torsion results, L={L}',clear=True)
    for uc,sol in res.items():
        if not np.isclose(L,sol.x[-1]):
            continue
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
    fig.tight_layout()
    plt.pause(plt_pause)
# %% solve
for L in (0.1,0.2,1,2):
    bvp(L)
    plot(L)
