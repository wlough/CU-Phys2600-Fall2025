import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton


#
#
# t0, u0, tf = 0, 1, 5
# Nt_arr = [20, 50, 100]
#
#
# def RHS_fun(t, u):
#     return u * (1 - np.cos(4 * t))
#
#
# def get_exact_fun(t0, u0):
#     C = u0 / np.exp(t0 - np.sin(4 * t0) / 4)
#     return lambda t: C * np.exp(t - np.sin(4 * t) / 4)
#
#
# t0, u0, tf = 0, 1, 5
# Nt_arr = [10, 20, 50]
#
#
# def RHS_fun(t, u):
#     return -u * (1 - np.cos(4 * t))
#
#
# def get_exact_fun(t0, u0):
#     C = u0 / np.exp(-t0 + np.sin(4 * t0) / 4)
#     return lambda t: C * np.exp(-t + np.sin(4 * t) / 4)
#
#
# t0, u0, tf = 0, 1, 5
# Nt_arr = [10, 15, 40]
#
#
# def RHS_fun(t, u):
#     return -3 * u
#
#
# def get_exact_fun(t0, u0):
#     C = u0 * np.exp(3 * t0)
#     return lambda t: C * np.exp(-3 * t)

# t0, u0, tf = 0, 0.0, 1
# Nt_arr = [10, 15, 40]

g = 9.8
m = 0.1
gamma = 1.5

u_inf = m * g / gamma
tau = m / gamma


def RHS_fun(t, u):
    return -(u - u_inf) / tau


def get_exact_fun(t0, u0):
    return lambda t: u0 * np.exp(-(t - t0) / tau) + u_inf * (
        1 - np.exp(-(t - t0) / tau)
    )


def be_implicit_eq(uip1, *args):
    ti, ui, dt = args
    tip1 = ti + dt
    return (uip1 - ui) / dt - RHS_fun(tip1, uip1)


def backward_euler_step(ti, ui, dt, uip1_guess=None):
    if uip1_guess is None:
        uip1_guess = ui
    uip1 = newton(be_implicit_eq, uip1_guess, args=(ti, ui, dt))
    # uip1, r = newton(be_implicit_eq, uip1_guess, args=(ti, ui, dt), full_output=True)
    # print(r.converged)
    return uip1


def forward_euler_step(ti, ui, dt):
    return ui + RHS_fun(ti, ui) * dt


def backward_euler_solve(t0, u0, tf, Nt):
    t_sol = np.linspace(t0, tf, Nt)
    u_sol = np.zeros_like(t_sol)
    u_sol[0] = u0
    dt = t_sol[1] - t_sol[0]
    for i in range(0, len(t_sol) - 1):
        u_sol[i + 1] = backward_euler_step(ti=t_sol[i], ui=u_sol[i], dt=dt)
    return t_sol, u_sol


def forward_euler_solve(t0, u0, tf, Nt):
    t_sol = np.linspace(t0, tf, Nt)
    u_sol = np.zeros_like(t_sol)
    u_sol[0] = u0
    dt = t_sol[1] - t_sol[0]
    for i in range(0, len(t_sol) - 1):
        u_sol[i + 1] = forward_euler_step(ti=t_sol[i], ui=u_sol[i], dt=dt)
    return t_sol, u_sol


def fb_compare_plot(t0, u0, tf, Nt_arr):

    u_fun = get_exact_fun(t0, u0)
    num_rows = len(Nt_arr)
    fig, axes = plt.subplots(1, num_rows, figsize=(8, 3.25))
    fig.suptitle(
        r"$u'(t)=-\left( u(t) - u_{\infty}\right)/\tau$",
        fontsize=18,
        x=1.15,
        y=0.5,
    )
    u_min = np.inf
    u_max = -np.inf
    for ax, Nt in zip(axes, Nt_arr):
        t_fesol, u_fesol = forward_euler_solve(t0, u0, tf, Nt)
        t_besol, u_besol = backward_euler_solve(t0, u0, tf, Nt)
        dt = t_fesol[1] - t_fesol[0]
        ax.set_title(r"$\Delta t =" + f"{dt: 0.2f}" + r"$")
        # u_min = np.min([u_min, *u_fesol, *u_besol])
        # u_max = np.max([u_max, *u_fesol, *u_besol])
        u_exact = u_fun(t_fesol)
        u_min = np.min([u_min, *u_fesol, *u_besol, *u_exact])
        u_max = np.max([u_max, *u_fesol, *u_besol, *u_exact])
        ax.plot(t_besol, u_besol, linestyle="--", label="Backward Euler")
        ax.plot(t_fesol, u_exact, linestyle="-", label="Exact")
        ax.plot(t_fesol, u_fesol, linestyle=":", label="Forward Euler")
        ax.grid(True)
        ax.legend(loc="lower right")
    # for ax in axes:
    #     ax.set_ylim(u_min, u_max)
    plt.tight_layout()
    plt.show()

