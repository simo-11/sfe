# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:35:37 2026

@author: simon
"""

import taichi as ti

ti.init(arch=ti.gpu if ti.cuda else ti.cpu)

# --- parametrit ---
n_board = 20          # laudan pisteet
board_length = 1.5
board_height = 0.5
dt = 1 / 240
substeps = 10

gravity = ti.Vector([0.0, -9.81])

board_mass = 1.0
jumper_mass = 70.0    # kokeile 30, 70, 120...

k_dist = 5e4          # pituusjäykkyys
k_bend = 1e3          # taipumajäykkyys
damping = 0.995

# --- data ---
x_board = ti.Vector.field(2, dtype=ti.f32, shape=n_board)
v_board = ti.Vector.field(2, dtype=ti.f32, shape=n_board)
inv_mass_board = ti.field(dtype=ti.f32, shape=n_board)

# hyppääjä = yksi piste
x_jumper = ti.Vector.field(2, dtype=ti.f32, shape=1)
v_jumper = ti.Vector.field(2, dtype=ti.f32, shape=1)
inv_mass_jumper = ti.field(dtype=ti.f32, shape=1)

# rest‑pituudet
rest_len = ti.field(dtype=ti.f32, shape=n_board - 1)
rest_bend = ti.field(dtype=ti.f32, shape=n_board - 2)


@ti.kernel
def init():
    # lauta vaakasuorassa, vasen pää kiinni
    for i in range(n_board):
        t = i / (n_board - 1)
        x_board[i] = ti.Vector([0.2 + board_length * t, board_height])
        v_board[i] = ti.Vector([0.0, 0.0])
        inv_mass_board[i] = 1.0 / (board_mass / n_board)
    inv_mass_board[0] = 0.0  # kiinnitetty pää

    for i in range(n_board - 1):
        rest_len[i] = (x_board[i + 1] - x_board[i]).norm()

    for i in range(n_board - 2):
        e1 = (x_board[i + 1] - x_board[i]).normalized()
        e2 = (x_board[i + 2] - x_board[i + 1]).normalized()
        rest_bend[i] = ti.acos(ti.min(0.9999, ti.max(-0.9999, e1.dot(e2))))

    # hyppääjä laudan yläpuolelle
    x_jumper[0] = ti.Vector([0.2 + board_length * 0.8, board_height + 0.6])
    v_jumper[0] = ti.Vector([0.0, 0.0])
    inv_mass_jumper[0] = 1.0 / jumper_mass


@ti.kernel
def apply_gravity():
    for i in range(n_board):
        if inv_mass_board[i] > 0:
            v_board[i] += gravity * dt
    for i in range(1):
        if inv_mass_jumper[i] > 0:
            v_jumper[i] += gravity * dt


@ti.kernel
def integrate():
    for i in range(n_board):
        x_board[i] += v_board[i] * dt
        v_board[i] *= damping
    for i in range(1):
        x_jumper[i] += v_jumper[i] * dt
        v_jumper[i] *= damping


@ti.kernel
def solve_distance():
    for i in range(n_board - 1):
        p1 = x_board[i]
        p2 = x_board[i + 1]
        w1 = inv_mass_board[i]
        w2 = inv_mass_board[i + 1]
        dir = p2 - p1
        L = dir.norm() + 1e-8
        C = L - rest_len[i]
        n = dir / L
        w_sum = w1 + w2
        if w_sum > 0:
            lambda_ = -k_dist * C / w_sum * dt * dt
            corr = lambda_ * n
            if w1 > 0:
                x_board[i] += corr * w1
            if w2 > 0:
                x_board[i + 1] -= corr * w2


@ti.kernel
def solve_bending():
    for i in range(n_board - 2):
        p0 = x_board[i]
        p1 = x_board[i + 1]
        p2 = x_board[i + 2]
        w0 = inv_mass_board[i]
        w1 = inv_mass_board[i + 1]
        w2 = inv_mass_board[i + 2]

        e1 = p1 - p0
        e2 = p2 - p1
        L1 = e1.norm() + 1e-8
        L2 = e2.norm() + 1e-8
        n1 = e1 / L1
        n2 = e2 / L2
        dot = ti.min(0.9999, ti.max(-0.9999, n1.dot(n2)))
        theta = ti.acos(dot)
        C = theta - rest_bend[i]

        # yksinkertainen kulmakorjaus
        axis = ti.Vector([-n1.y, n1.x])  # 2D "normaali"
        grad0 = axis
        grad2 = -axis
        grad1 = -(grad0 + grad2)

        w_sum = w0 * grad0.norm_sqr() + w1 * grad1.norm_sqr() + w2 * grad2.norm_sqr()
        if w_sum > 0:
            lambda_ = -k_bend * C / w_sum * dt * dt
            if w0 > 0:
                x_board[i] += lambda_ * w0 * grad0
            if w1 > 0:
                x_board[i + 1] += lambda_ * w1 * grad1
            if w2 > 0:
                x_board[i + 2] += lambda_ * w2 * grad2


@ti.kernel
def solve_contact():
    # yksinkertainen kontakti: hyppääjä ei saa mennä laudan "läpi"
    for i in range(n_board - 1):
        p1 = x_board[i]
        p2 = x_board[i + 1]
        seg = p2 - p1
        seg_len = seg.norm() + 1e-8
        t = ((x_jumper[0] - p1).dot(seg) / seg_len**2)
        t = ti.min(1.0, ti.max(0.0, t))
        closest = p1 + t * seg
        n = ti.Vector([-seg.y, seg.x]).normalized()
        # oletetaan, että lauta "ylöspäin" = n
        rel = x_jumper[0] - closest
        dist = rel.dot(n)
        radius = 0.03
        if dist < radius:
            # korjaa tunkeuma
            wj = inv_mass_jumper[0]
            w1 = inv_mass_board[i]
            w2 = inv_mass_board[i + 1]
            w_sum = wj + w1 + w2
            if w_sum > 0:
                corr = (radius - dist) * n
                if wj > 0:
                    x_jumper[0] += corr * (wj / w_sum)
                if w1 > 0:
                    x_board[i] -= corr * (w1 / w_sum)
                if w2 > 0:
                    x_board[i + 1] -= corr * (w2 / w_sum)


window = ti.ui.Window("Springboard XPBD", (800, 600))
canvas = window.get_canvas()
init()

while window.running:
    for _ in range(substeps):
        apply_gravity()
        integrate()
        solve_distance()
        solve_bending()
        solve_contact()

    canvas.clear(0x112233)
    # piirrään laudan
    board_pts = x_board.to_numpy()
    for i in range(n_board - 1):
        a = board_pts[i]
        b = board_pts[i + 1]
        canvas.lines(a, b, radius=1.5, color=0xEEEE55)

    # hyppääjä
    j = x_jumper.to_numpy()[0]
    canvas.circles(j[None, :], radius=6, color=0xFF5555)

    window.show()
