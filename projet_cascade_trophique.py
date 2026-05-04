import os
import time
import numpy as np
import matplotlib.pyplot as plt



# PARAMÈTRES DU SUJET 


# Végétation
rV = 1.20
KV = 500.0

aN = 15.0
vN = 200.0

aD = 8.0
vD = 100.0

aB = 3.0
vB = 120.0

# Ongulés principaux
rN = 0.30
KN = 7.0
Vetoile = 70.0
thetaN = 4.0
cWN = 7.5
b = 0.5
cBN = 0.5
hBN = 3.0
dN = 0.05

# Ongulé secondaire
rD = 0.50
KD = 4.0
Vetoile_etoile = 50.0
thetaD = 2.0
cWD = 2.0
hWD = 2.0
cBD = 0.3
hBD = 1.5
dD = 0.08

# Loups
e1 = 0.25
e2 = 0.5
n = 0.01
mW = 0.15
muW = 0.10

# Ours
eB = 0.12
eV = 0.02
mB = 0.08



# CONDITIONS I


V0 = 440.0
N0 = 4.5
D0 = 2.5
W0 = 0.04
B0 = 0.25
u0 = np.array([V0, N0, D0, W0, B0])
t0 = 0.0
tf = 50.0
h = 0.05




def F(t, u):
    V, N, D, W, B = u

    kappaN = KN * V / (Vetoile + V)
    kappaD = KD * V / (Vetoile_etoile + V)

    phiW = cWN * N / (W + b * N) + cWD * D / (hWD + D)

    dV_dt = rV * V * (1.0 - V / KV) - aN * V * N / (vN + V) - aD * V * D / (vD + V) - aB * V * B / (vB + V)

    dN_dt = rN * N * (1.0 - (N / kappaN) ** thetaN) - cWN * N * W / (W + b * N) - cBN * N * B / (hBN + N) - dN * N

    dD_dt = rD * D * (1.0 - (D / kappaD) ** thetaD) - cWD * D * W / (hWD + D) - cBD * D * B / (hBD + D) - dD * D

    dW_dt = (e1 * np.log(phiW + n) - e2) * W - mW * W - muW * (np.sin(np.pi * t) ** 2) * W

    dB_dt = (eB * (cBN * N / (hBN + N) + cBD * D / (hBD + D)) + eV * aB * V / (vB + V) - mB) * B

    return np.array([dV_dt, dN_dt, dD_dt, dW_dt, dB_dt])



def euler_explicite(u0, t0, tf, h):
    n = int((tf - t0) / h)
    t = np.linspace(t0, tf, n + 1)

    U = np.zeros((n + 1, 5))
    U[0] = u0

    for i in range(n):
        U[i + 1] = U[i] + h * F(t[i], U[i])

    return t, U




def rk4(u0, t0, tf, h):
    n = int((tf - t0) / h)
    t = np.linspace(t0, tf, n + 1)

    U = np.zeros((n + 1, 5))
    U[0] = u0

    for i in range(n):
        k1 = F(t[i], U[i])
        k2 = F(t[i] + h / 2, U[i] + h * k1 / 2)
        k3 = F(t[i] + h / 2, U[i] + h * k2 / 2)
        k4 = F(t[i] + h, U[i] + h * k3)

        U[i + 1] = U[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t, U





def decomposition_lu(A):
    A = A.copy().astype(float)
    n = len(A)

    L = np.eye(n)
    U = A.copy()
    P = np.eye(n)

    for k in range(n - 1):
        pivot = k + np.argmax(np.abs(U[k:, k]))

        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            P[[k, pivot]] = P[[pivot, k]]

            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i] = U[i] - L[i, k] * U[k]

    return P, L, U


def descente(L, b):
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        s = 0.0

        for j in range(i):
            s = s + L[i, j] * y[j]

        y[i] = (b[i] - s) / L[i, i]

    return y


def remontee(U, y):
    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        s = 0.0

        for j in range(i + 1, n):
            s = s + U[i, j] * x[j]

        x[i] = (y[i] - s) / U[i, i]

    return x


def resolution_lu(A, b):
    P, L, U = decomposition_lu(A)
    y = descente(L, P @ b)
    x = remontee(U, y)

    return x




def jacobienne_numerique(G, x):
    eps = 1e-6
    n = len(x)

    J = np.zeros((n, n))

    for j in range(n):
        x_plus = x.copy()
        x_moins = x.copy()

        x_plus[j] = x_plus[j] + eps
        x_moins[j] = x_moins[j] - eps

        J[:, j] = (G(x_plus) - G(x_moins)) / (2 * eps)

    return J





def newton(G, x0, tol=1e-9, max_iter=20):
    x = x0.copy()

    for k in range(max_iter):
        gx = G(x)

        if np.linalg.norm(gx) < tol:
            return x

        J = jacobienne_numerique(G, x)
        delta = resolution_lu(J, -gx)

        x = x + delta

    return x



def euler_implicite_newton(u0, t0, tf, h):
    n = int((tf - t0) / h)
    t = np.linspace(t0, tf, n + 1)

    U = np.zeros((n + 1, 5))
    U[0] = u0

    for i in range(n):
        un = U[i].copy()
        tn1 = t[i + 1]

        def G(x):
            return x - un - h * F(tn1, x)

        # Point de départ : Euler explicite
        x_depart = un + h * F(t[i], un)

        x = newton(G, x_depart)

        U[i + 1] = x

    return t, U



def euler_implicite_point_fixe(u0, t0, tf, h, tol=1e-9, max_iter=200):
    n = int((tf - t0) / h)
    t = np.linspace(t0, tf, n + 1)

    U = np.zeros((n + 1, 5))
    U[0] = u0

    for i in range(n):
        un = U[i].copy()
        tn1 = t[i + 1]

        x = un + h * F(t[i], un)

        for k in range(max_iter):
            x_new = un + h * F(tn1, x)

            if np.linalg.norm(x_new - x) < tol:
                x = x_new
                break

            x = x_new

        U[i + 1] = x

    return t, U


if __name__ == "__main__":

    dossier = "figures"
    os.makedirs(dossier, exist_ok=True)

    debut = time.perf_counter()
    t_euler, U_euler = euler_explicite(u0, t0, tf, h)
    temps_euler = time.perf_counter() - debut

    debut = time.perf_counter()
    t_rk4, U_rk4 = rk4(u0, t0, tf, h)
    temps_rk4 = time.perf_counter() - debut

    debut = time.perf_counter()
    t_pf, U_pf = euler_implicite_point_fixe(u0, t0, tf, h)
    temps_pf = time.perf_counter() - debut

    debut = time.perf_counter()
    t_base, U_base = euler_implicite_newton(u0, t0, tf, h)
    temps_newton = time.perf_counter() - debut


    # 1) Scénario de base

    plt.figure(figsize=(8, 4))
    plt.plot(t_base, U_base[:, 0])
    plt.xlabel("Temps")
    plt.ylabel("V(t)")
    plt.title("Scénario de base : végétation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "base_vegetation.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_base, U_base[:, 1], label="N(t)")
    plt.plot(t_base, U_base[:, 2], label="D(t)")
    plt.plot(t_base, U_base[:, 3], label="W(t)")
    plt.plot(t_base, U_base[:, 4], label="B(t)")
    plt.xlabel("Temps")
    plt.ylabel("Population")
    plt.title("Scénario de base : populations animales")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "base_animaux.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_base, U_base[:, 1], label="N(t)")
    plt.plot(t_base, U_base[:, 2], label="D(t)")
    plt.xlabel("Temps")
    plt.ylabel("Population")
    plt.title("Scénario de base : ongulés")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "base_ongules.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_base, U_base[:, 3], label="W(t)")
    plt.plot(t_base, U_base[:, 4], label="B(t)")
    plt.xlabel("Temps")
    plt.ylabel("Population")
    plt.title("Scénario de base : prédateurs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "base_predateurs.png"), dpi=180)
    plt.close()


    # 2) Comparaison des méthodes avec h = 0.05

    plt.figure(figsize=(8, 4))
    plt.plot(t_euler, U_euler[:, 1], label="Euler explicite")
    plt.plot(t_rk4, U_rk4[:, 1], label="RK4")
    plt.plot(t_pf, U_pf[:, 1], label="Euler implicite point fixe")
    plt.plot(t_base, U_base[:, 1], label="Euler implicite Newton")
    plt.xlabel("Temps")
    plt.ylabel("N(t)")
    plt.title("Comparaison des méthodes avec h = 0.05 : N(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "comparaison_methodes_h005_N.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_euler, U_euler[:, 2], label="Euler explicite")
    plt.plot(t_rk4, U_rk4[:, 2], label="RK4")
    plt.plot(t_pf, U_pf[:, 2], label="Euler implicite point fixe")
    plt.plot(t_base, U_base[:, 2], label="Euler implicite Newton")
    plt.xlabel("Temps")
    plt.ylabel("D(t)")
    plt.title("Comparaison des méthodes avec h = 0.05 : D(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "comparaison_methodes_h005_D.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_euler, U_euler[:, 3], label="Euler explicite")
    plt.plot(t_rk4, U_rk4[:, 3], label="RK4")
    plt.plot(t_pf, U_pf[:, 3], label="Euler implicite point fixe")
    plt.plot(t_base, U_base[:, 3], label="Euler implicite Newton")
    plt.xlabel("Temps")
    plt.ylabel("W(t)")
    plt.title("Comparaison des méthodes avec h = 0.05 : W(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "comparaison_methodes_h005_W.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_euler, U_euler[:, 4], label="Euler explicite")
    plt.plot(t_rk4, U_rk4[:, 4], label="RK4")
    plt.plot(t_pf, U_pf[:, 4], label="Euler implicite point fixe")
    plt.plot(t_base, U_base[:, 4], label="Euler implicite Newton")
    plt.xlabel("Temps")
    plt.ylabel("B(t)")
    plt.title("Comparaison des méthodes avec h = 0.05 : B(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "comparaison_methodes_h005_B.png"), dpi=180)
    plt.close()


    # 3) Comparaison des méthodes avec h = 0.5

    h_grand = 0.5

    t_euler_hg, U_euler_hg = euler_explicite(u0, t0, tf, h_grand)
    t_rk4_hg, U_rk4_hg = rk4(u0, t0, tf, h_grand)
    t_pf_hg, U_pf_hg = euler_implicite_point_fixe(u0, t0, tf, h_grand)
    t_newton_hg, U_newton_hg = euler_implicite_newton(u0, t0, tf, h_grand)

    plt.figure(figsize=(8, 4))
    plt.plot(t_euler_hg, U_euler_hg[:, 1], label="Euler explicite")
    plt.plot(t_rk4_hg, U_rk4_hg[:, 1], label="RK4")
    plt.plot(t_pf_hg, U_pf_hg[:, 1], label="Euler implicite point fixe")
    plt.plot(t_newton_hg, U_newton_hg[:, 1], label="Euler implicite Newton")
    plt.xlabel("Temps")
    plt.ylabel("N(t)")
    plt.title("Comparaison des méthodes avec h = 0.5 : N(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "comparaison_methodes_h05_N.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_euler_hg, U_euler_hg[:, 2], label="Euler explicite")
    plt.plot(t_rk4_hg, U_rk4_hg[:, 2], label="RK4")
    plt.plot(t_pf_hg, U_pf_hg[:, 2], label="Euler implicite point fixe")
    plt.plot(t_newton_hg, U_newton_hg[:, 2], label="Euler implicite Newton")
    plt.xlabel("Temps")
    plt.ylabel("D(t)")
    plt.title("Comparaison des méthodes avec h = 0.5 : D(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "comparaison_methodes_h05_D.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_euler_hg, U_euler_hg[:, 3], label="Euler explicite")
    plt.plot(t_rk4_hg, U_rk4_hg[:, 3], label="RK4")
    plt.plot(t_pf_hg, U_pf_hg[:, 3], label="Euler implicite point fixe")
    plt.plot(t_newton_hg, U_newton_hg[:, 3], label="Euler implicite Newton")
    plt.xlabel("Temps")
    plt.ylabel("W(t)")
    plt.title("Comparaison des méthodes avec h = 0.5 : W(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "comparaison_methodes_h05_W.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_euler_hg, U_euler_hg[:, 4], label="Euler explicite")
    plt.plot(t_rk4_hg, U_rk4_hg[:, 4], label="RK4")
    plt.plot(t_pf_hg, U_pf_hg[:, 4], label="Euler implicite point fixe")
    plt.plot(t_newton_hg, U_newton_hg[:, 4], label="Euler implicite Newton")
    plt.xlabel("Temps")
    plt.ylabel("B(t)")
    plt.title("Comparaison des méthodes avec h = 0.5 : B(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "comparaison_methodes_h05_B.png"), dpi=180)
    plt.close()


    # 4) Influence du pas de temps sur W(t)  montrer divergence h entier

    t_h2, U_h2 = euler_implicite_newton(u0, t0, tf, 2.0)
    t_h15, U_h15 = euler_implicite_newton(u0, t0, tf, 1.5)
    t_h1, U_h1 = euler_implicite_newton(u0, t0, tf, 1.0)
    t_h05, U_h05 = euler_implicite_newton(u0, t0, tf, 0.5)

    plt.figure(figsize=(8, 4))
    plt.plot(t_h2, U_h2[:, 3], label="h = 2")
    plt.plot(t_h15, U_h15[:, 3], label="h = 1.5")
    plt.plot(t_h1, U_h1[:, 3], label="h = 1")
    plt.plot(t_h05, U_h05[:, 3], label="h = 0.5")
    plt.xlabel("Temps")
    plt.ylabel("W(t)")
    plt.title("Influence de grands pas de temps sur W(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "pas_temps_W_divergence.png"), dpi=180)
    plt.close()

    t_h2, U_h2 = euler_implicite_newton(u0, t0, tf, 2.0)
    t_h15, U_h15 = euler_implicite_newton(u0, t0, tf, 1.5)
    t_h1, U_h1 = euler_implicite_newton(u0, t0, tf, 1.0)

    plt.figure(figsize=(8, 4))
    plt.plot(t_h2, U_h2[:, 1], label="h = 2")
    plt.plot(t_h15, U_h15[:, 1], label="h = 1.5")
    plt.plot(t_h1, U_h1[:, 1], label="h = 1")
    plt.xlabel("Temps")
    plt.ylabel("N(t)")
    plt.title("Influence de grands pas de temps sur N(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "pas_temps_N_grands_pas.png"), dpi=180)
    plt.close()


    # Comparaison des méthodes avec h = 1.5 sur N(t)

    h_test = 1.5

    t_euler_h15, U_euler_h15 = euler_explicite(u0, t0, tf, h_test)
    t_rk4_h15, U_rk4_h15 = rk4(u0, t0, tf, h_test)
    t_pf_h15, U_pf_h15 = euler_implicite_point_fixe(u0, t0, tf, h_test)
    t_newton_h15, U_newton_h15 = euler_implicite_newton(u0, t0, tf, h_test)

    plt.figure(figsize=(8, 4))
    plt.plot(t_euler_h15, U_euler_h15[:, 1], label="Euler explicite")
    plt.plot(t_rk4_h15, U_rk4_h15[:, 1], label="RK4")
    plt.plot(t_pf_h15, U_pf_h15[:, 1], linewidth=2.2, label="Euler implicite point fixe")
    plt.plot(t_newton_h15, U_newton_h15[:, 1], linewidth=2.5, label="Euler implicite Newton")
    plt.xlabel("Temps")
    plt.ylabel("N(t)")
    plt.title("Comparaison des méthodes avec h = 1.5 : N(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "comparaison_methodes_h15_N.png"), dpi=180)
    plt.close()


    # 5) Effet du prélèvement humain muW

    ancien_muW = muW

    muW = 0.0
    t_mu0, U_mu0 = euler_implicite_newton(u0, t0, tf, h)

    muW = 0.10
    t_mu01, U_mu01 = euler_implicite_newton(u0, t0, tf, h)

    muW = 0.20
    t_mu02, U_mu02 = euler_implicite_newton(u0, t0, tf, h)

    muW = 0.40
    t_mu04, U_mu04 = euler_implicite_newton(u0, t0, tf, h)

    muW = ancien_muW

    plt.figure(figsize=(8, 4))
    plt.plot(t_mu0, U_mu0[:, 0], label="muW = 0.0")
    plt.plot(t_mu01, U_mu01[:, 0], label="muW = 0.1")
    plt.plot(t_mu02, U_mu02[:, 0], label="muW = 0.2")
    plt.plot(t_mu04, U_mu04[:, 0], label="muW = 0.4")
    plt.xlabel("Temps")
    plt.ylabel("V(t)")
    plt.title("Effet du prélèvement sur V(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "prelevement_V.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_mu0, U_mu0[:, 1], label="muW = 0.0")
    plt.plot(t_mu01, U_mu01[:, 1], label="muW = 0.1")
    plt.plot(t_mu02, U_mu02[:, 1], label="muW = 0.2")
    plt.plot(t_mu04, U_mu04[:, 1], label="muW = 0.4")
    plt.xlabel("Temps")
    plt.ylabel("N(t)")
    plt.title("Effet du prélèvement sur N(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "prelevement_N.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_mu0, U_mu0[:, 2], label="muW = 0.0")
    plt.plot(t_mu01, U_mu01[:, 2], label="muW = 0.1")
    plt.plot(t_mu02, U_mu02[:, 2], label="muW = 0.2")
    plt.plot(t_mu04, U_mu04[:, 2], label="muW = 0.4")
    plt.xlabel("Temps")
    plt.ylabel("D(t)")
    plt.title("Effet du prélèvement sur D(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "prelevement_D.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_mu0, U_mu0[:, 3], label="muW = 0.0")
    plt.plot(t_mu01, U_mu01[:, 3], label="muW = 0.1")
    plt.plot(t_mu02, U_mu02[:, 3], label="muW = 0.2")
    plt.plot(t_mu04, U_mu04[:, 3], label="muW = 0.4")
    plt.xlabel("Temps")
    plt.ylabel("W(t)")
    plt.title("Effet du prélèvement sur W(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "prelevement_W.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_mu0, U_mu0[:, 4], label="muW = 0.0")
    plt.plot(t_mu01, U_mu01[:, 4], label="muW = 0.1")
    plt.plot(t_mu02, U_mu02[:, 4], label="muW = 0.2")
    plt.plot(t_mu04, U_mu04[:, 4], label="muW = 0.4")
    plt.xlabel("Temps")
    plt.ylabel("B(t)")
    plt.title("Effet du prélèvement sur B(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "prelevement_B.png"), dpi=180)
    plt.close()


    # 6) Scénario Yellowstone

    u0_yellowstone = np.array([V0, N0, D0, 0.0001, B0])

    t1, U1 = euler_implicite_newton(u0_yellowstone, 0.0, 10.0, h)

    u_reintro = U1[-1].copy()
    u_reintro[3] = 0.04

    t2, U2 = euler_implicite_newton(u_reintro, 10.0, 50.0, h)

    t_y = np.concatenate((t1, t2[1:]))
    U_y = np.vstack((U1, U2[1:]))

    plt.figure(figsize=(8, 4))
    plt.plot(t_base, U_base[:, 0], label="Scénario de base")
    plt.plot(t_y, U_y[:, 0], "--", label="Yellowstone")
    plt.axvline(10.0, color="black", linestyle=":", label="Réintroduction")
    plt.xlabel("Temps")
    plt.ylabel("V(t)")
    plt.title("Yellowstone vs base : V(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "yellowstone_V.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_base, U_base[:, 1], label="Scénario de base")
    plt.plot(t_y, U_y[:, 1], "--", label="Yellowstone")
    plt.axvline(10.0, color="black", linestyle=":", label="Réintroduction")
    plt.xlabel("Temps")
    plt.ylabel("N(t)")
    plt.title("Yellowstone vs base : N(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "yellowstone_N.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_base, U_base[:, 2], label="Scénario de base")
    plt.plot(t_y, U_y[:, 2], "--", label="Yellowstone")
    plt.axvline(10.0, color="black", linestyle=":", label="Réintroduction")
    plt.xlabel("Temps")
    plt.ylabel("D(t)")
    plt.title("Yellowstone vs base : D(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "yellowstone_D.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_base, U_base[:, 3], label="Scénario de base")
    plt.plot(t_y, U_y[:, 3], "--", label="Yellowstone")
    plt.axvline(10.0, color="black", linestyle=":", label="Réintroduction")
    plt.xlabel("Temps")
    plt.ylabel("W(t)")
    plt.title("Yellowstone vs base : W(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "yellowstone_W.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_base, U_base[:, 4], label="Scénario de base")
    plt.plot(t_y, U_y[:, 4], "--", label="Yellowstone")
    plt.axvline(10.0, color="black", linestyle=":", label="Réintroduction")
    plt.xlabel("Temps")
    plt.ylabel("B(t)")
    plt.title("Yellowstone vs base : B(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "yellowstone_B.png"), dpi=180)
    plt.close()


     # 7) Affichage console

    print("Temps de calcul")
    print("Euler explicite :", round(temps_euler, 4), "s")
    print("RK4 :", round(temps_rk4, 4), "s")
    print("Euler implicite point fixe :", round(temps_pf, 4), "s")
    print("Euler implicite Newton + LU :", round(temps_newton, 4), "s")
