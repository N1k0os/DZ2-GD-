import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import griddata
import matplotlib as mpl
# погнал вайбкодинг небольшой <3

mpl.rcParams['figure.figsize'] = (14, 10)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.weight'] = 'normal'
mpl.rcParams['font.size'] = 18
mpl.rcParams['font.style'] = 'normal'
mpl.rcParams['mathtext.default'] = 'regular'

# --- ИСХОДНЫЕ ДАННЫЕ ДЛЯ ВАРИАНТА 18 ---
VARIANT = 18
M1 = 4.0  # число Маха набегающего потока
beta1 = np.radians(7.9)  # угол 1-го клина
beta2 = np.radians(17.4)  # угол 2-го клина
beta3 = np.radians(30.0)  # угол 3-го клина
k = 1.4  # показатель адиабаты


# --- ФУНКЦИИ РАСЧЕТА ---
def M_out(M_in, nu, beta, k):
    Msisq = (M_in * np.sin(nu)) ** 2
    km1 = k - 1
    numer = np.sqrt(1 + 0.5 * km1 * Msisq)
    denom = np.sin(nu - beta) * np.sqrt(k * Msisq - 0.5 * km1)
    return numer / denom


def p_rel(M_in, nu, k):
    Msisq = (M_in * np.sin(nu)) ** 2
    return 2 * k * Msisq / (k + 1) - (k - 1) / (k + 1)


def rho_rel(M_in, nu, k):
    Msisq = (M_in * np.sin(nu)) ** 2
    return 0.5 * (k + 1) * Msisq / (1 + 0.5 * (k - 1) * Msisq)


def T_rel(M_in, nu, k):
    Msisq = (M_in * np.sin(nu)) ** 2
    km1 = k - 1
    kp1 = k + 1
    factor_1 = 2 * k * Msisq / km1 - 1
    factor_2 = (2 / (km1 * Msisq) + 1) * (km1 / kp1) ** 2
    return factor_1 * factor_2


def sigma(M_in, nu, k):
    Msisq = (M_in * np.sin(nu)) ** 2
    km1 = k - 1
    kp1 = k + 1
    numer = (0.5 * kp1) ** (kp1 / km1)
    denom_f1 = (1 / Msisq + 0.5 * km1) ** (k / km1)
    denom_f2 = (k * Msisq - 0.5 * km1) ** (1 / km1)
    return numer / (denom_f1 * denom_f2)


def f(nu, M_in, beta, k):
    kp1 = k + 1
    A = np.sin(nu) ** 2 - 1 / M_in ** 2
    return np.tan(beta) - A / (np.tan(nu) * (0.5 * kp1 - A))


def bisection_method(M_in, beta, k, a, b, tol):
    if f(a, M_in, beta, k) * f(b, M_in, beta, k) > 0:
        return None
    while (b - a) / 2 > tol:
        midpoint = (a + b) / 2
        if f(midpoint, M_in, beta, k) == 0:
            return midpoint
        elif f(a, M_in, beta, k) * f(midpoint, M_in, beta, k) < 0:
            b = midpoint
        else:
            a = midpoint
    return midpoint


def find_nu(M_in, beta, k):
    return bisection_method(M_in, beta, k, 0.01, 1.2, 1e-15)


def focus(x_f, y_f, x_p, y_p, beta, theta):
    numer = y_f - x_f * np.tan(theta) - (y_p - x_p * np.tan(beta))
    denom = np.tan(beta) - np.tan(theta)
    x_n = numer / denom
    y_n = y_p + (x_n - x_p) * np.tan(beta)
    return x_n, y_n


def sigma_total(M_p, beta_list, k):
    M_cur = M_p
    beta_prev = 0
    sgm_t = 1
    for i in range(len(beta_list)):
        d_beta = beta_list[i] - beta_prev
        nu = find_nu(M_cur, d_beta, k)
        if nu is None:
            return None
        sgm_t *= sigma(M_cur, nu, k)
        M_cur = M_out(M_cur, nu, d_beta, k)
        beta_prev = beta_list[i]
    return sgm_t


# --- РАСЧЕТ ТЕЧЕНИЯ ---
dbeta1 = beta1
dbeta2 = beta2 - beta1
dbeta3 = beta3 - beta2

nu1 = find_nu(M1, dbeta1, k)
theta1 = nu1
M2 = M_out(M1, nu1, dbeta1, k)
sigma1 = sigma(M1, nu1, k)

nu2 = find_nu(M2, dbeta2, k)
theta2 = nu2 + beta1
M3 = M_out(M2, nu2, dbeta2, k)
sigma2 = sigma(M2, nu2, k)

nu3 = find_nu(M3, dbeta3, k)
theta3 = nu3 + beta2
M4 = M_out(M3, nu3, dbeta3, k)
sigma3 = sigma(M3, nu3, k)

sigma_sum = sigma1 * sigma2 * sigma3

# --- ФОКУСИРОВКА ИСХОДНОГО КЛИНА ---
yf = 1.0
xf = yf / np.tan(nu1)
x1, y1 = 0, 0
x2, y2 = focus(xf, yf, x1, y1, beta1, theta2)
x3, y3 = focus(xf, yf, x2, y2, beta2, theta3)


# --- ОПТИМИЗАЦИЯ ---
def objective(X):
    b_list = [X[0], X[1], beta3]
    res = sigma_total(M1, b_list, k)
    return -res if res is not None else 1e6


constraints = [
    {'type': 'ineq', 'fun': lambda X: X[0] - 1e-4},
    {'type': 'ineq', 'fun': lambda X: beta3 - X[0]},
    {'type': 'ineq', 'fun': lambda X: X[1] - X[0] - 1e-4},
    {'type': 'ineq', 'fun': lambda X: beta3 - X[1]},
]
bounds = [(1e-4, beta3), (1e-4, beta3)]
res = minimize(objective, x0=[beta1, beta2], bounds=bounds, constraints=constraints, method='SLSQP')
opt_beta1, opt_beta2 = res.x[0], res.x[1]
sigma_sum_max = -res.fun

# --- ФОКУСИРОВКА ОПТИМИЗИРОВАННОГО КЛИНА ---
opt_dbeta1 = opt_beta1
opt_dbeta2 = opt_beta2 - opt_beta1
opt_dbeta3 = beta3 - opt_beta2

opt_nu1 = find_nu(M1, opt_dbeta1, k)
opt_theta1 = opt_nu1
opt_M2 = M_out(M1, opt_nu1, opt_dbeta1, k)

opt_nu2 = find_nu(opt_M2, opt_dbeta2, k)
opt_theta2 = opt_nu2 + opt_beta1
opt_M3 = M_out(opt_M2, opt_nu2, opt_dbeta2, k)

opt_nu3 = find_nu(opt_M3, opt_dbeta3, k)
opt_theta3 = opt_nu3 + opt_beta2

xf_opt = 1.0 / np.tan(opt_nu1)
x2_opt, y2_opt = focus(xf_opt, 1.0, 0, 0, opt_beta1, opt_theta2)
x3_opt, y3_opt = focus(xf_opt, 1.0, x2_opt, y2_opt, opt_beta2, opt_theta3)

# ==========================================
#             ПОСТРОЕНИЕ ГРАФИКОВ
# ==========================================

# Длина продолжения клина для красивого отображения
length_extension = 0.8

# --- ГРАФИК 1: Геометрия исходного клина ---
fig1, ax1 = plt.subplots(figsize=(14, 10))

# Клин (красные линии)
x_end = x3 + length_extension * np.cos(beta3)
y_end = y3 + length_extension * np.sin(beta3)
ax1.plot([x1, x2], [y1, y2], 'r-', linewidth=3, label='Клин')
ax1.plot([x2, x3], [y2, y3], 'r-', linewidth=3)
ax1.plot([x3, x_end], [y3, y_end], 'r-', linewidth=3)

# Система КСУ (зеленые линии)
ax1.plot([0, xf], [0, 1], 'g-', linewidth=2, label='КСУ')
ax1.plot([x2, xf], [y2, 1], 'g-', linewidth=2)
ax1.plot([x3, xf], [y3, 1], 'g-', linewidth=2)

# Точки излома (черные маркеры)
ax1.plot([0, x2, x3], [0, y2, y3], 'ko', markersize=8, label='Точки излома')

ax1.set_xlabel('x, м', fontsize=16)
ax1.set_ylabel('y, м', fontsize=16)
ax1.set_title(f'Геометрия трехступенчатого клина и система КСУ\n'
              f'β₁={np.degrees(beta1):.1f}°, β₂={np.degrees(beta2):.1f}°, β₃={np.degrees(beta3):.1f}°, M₁={M1}',
              fontsize=18)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=14, loc='upper left')
ax1.set_aspect('equal', adjustable='box')

ax1.text(0, 0, '  (0,0)', fontsize=12, verticalalignment='bottom')
ax1.text(x2, y2, f'  ({x2:.3f},{y2:.3f})', fontsize=12, verticalalignment='bottom')
ax1.text(x3, y3, f'  ({x3:.3f},{y3:.3f})', fontsize=12, verticalalignment='bottom')
ax1.text(xf, 1, '  F', fontsize=12, verticalalignment='bottom')

ax1.set_xlim(-0.1, max(x_end, xf) + 0.2)
ax1.set_ylim(-0.1, max(y_end, 1) + 0.2)

plt.tight_layout()
plt.show()

# --- ГРАФИК 2: Контурный график поля поиска ---
N = 100
b1_arr = np.linspace(0.01, beta3, N)
b1_res, b2_res, sig_res = [], [], []

for i in range(len(b1_arr)):
    b2_arr = np.linspace(b1_arr[i], beta3, N)
    for j in range(len(b2_arr)):
        s = sigma_total(M1, [b1_arr[i], b2_arr[j], beta3], k)
        if s is not None:
            b1_res.append(b1_arr[i])
            b2_res.append(b2_arr[j])
            sig_res.append(s)

grid_b1, grid_b2 = np.meshgrid(np.linspace(0.01, beta3, N), np.linspace(0.01, beta3, N))
grid_sig = griddata((b1_res, b2_res), sig_res, (grid_b1, grid_b2), method='linear')

# Маскировка области, где beta1 > beta2 (недопустимая зона)
mask = grid_b1 > grid_b2
grid_sig_masked = np.ma.masked_where(mask, grid_sig)

fig2, ax2 = plt.subplots(figsize=(12, 8))
cf = ax2.contourf(grid_b1, grid_b2, grid_sig_masked, levels=100, cmap='jet')
cbar = plt.colorbar(cf, ax=ax2)
cbar.set_label(r'$\sigma_{\Sigma}$', fontsize=16)

ax2.plot(beta1, beta2, 'g^', markersize=15, label='Исходные углы')
ax2.plot(opt_beta1, opt_beta2, 'y*', markersize=18, label='Оптимальные углы')

ax2.set_xlabel(r'$\beta_1$, рад', fontsize=16)
ax2.set_ylabel(r'$\beta_2$, рад', fontsize=16)
ax2.set_title(r'Поле поиска оптимального значения $\sigma_{\Sigma}$', fontsize=18)
ax2.set_xlim(0, beta3)
ax2.set_ylim(0, beta3)
ax2.legend(fontsize=14)

plt.tight_layout()
plt.show()

# --- ГРАФИК 3: Геометрия оптимального клина ---
fig3, ax3 = plt.subplots(figsize=(14, 10))

# Клин (красные линии)
x_end_opt = x3_opt + length_extension * np.cos(beta3)
y_end_opt = y3_opt + length_extension * np.sin(beta3)
ax3.plot([0, x2_opt], [0, y2_opt], 'r-', linewidth=3, label='Клин')
ax3.plot([x2_opt, x3_opt], [y2_opt, y3_opt], 'r-', linewidth=3)
ax3.plot([x3_opt, x_end_opt], [y3_opt, y_end_opt], 'r-', linewidth=3)

# Система КСУ (зеленые линии)
ax3.plot([0, xf_opt], [0, 1], 'g-', linewidth=2, label='КСУ')
ax3.plot([x2_opt, xf_opt], [y2_opt, 1], 'g-', linewidth=2)
ax3.plot([x3_opt, xf_opt], [y3_opt, 1], 'g-', linewidth=2)

# Точки излома (черные маркеры)
ax3.plot([0, x2_opt, x3_opt], [0, y2_opt, y3_opt], 'ko', markersize=8, label='Точки излома')

ax3.set_xlabel('x, м', fontsize=16)
ax3.set_ylabel('y, м', fontsize=16)
ax3.set_title(f'Решение задачи фокусировки для оптимальных значений углов\n'
              f'β₁={np.degrees(opt_beta1):.1f}°, β₂={np.degrees(opt_beta2):.1f}°, β₃={np.degrees(beta3):.1f}°, M₁={M1}',
              fontsize=18)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=14, loc='upper left')
ax3.set_aspect('equal', adjustable='box')

ax3.text(0, 0, '  (0,0)', fontsize=12, verticalalignment='bottom')
ax3.text(x2_opt, y2_opt, f'  ({x2_opt:.3f},{y2_opt:.3f})', fontsize=12, verticalalignment='bottom')
ax3.text(x3_opt, y3_opt, f'  ({x3_opt:.3f},{y3_opt:.3f})', fontsize=12, verticalalignment='bottom')
ax3.text(xf_opt, 1, '  F', fontsize=12, verticalalignment='bottom')

ax3.set_xlim(-0.1, max(x_end_opt, xf_opt) + 0.2)
ax3.set_ylim(-0.1, max(y_end_opt, 1) + 0.2)

plt.tight_layout()
plt.show()


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
