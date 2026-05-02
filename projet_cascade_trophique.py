import os
import time
import numpy as np
import matplotlib.pyplot as plt



# PARAMÈTRES DU MODÈLE


# Végétation
rV = 1.20
KV = 500.0
alphaN = 15.0
nuN = 200.0
alphaD = 8.0
nuD = 100.0
alphaB = 3.0
nuB = 120.0

# Ongulés principaux
rN = 0.30
KN = 7.0
VstarN = 70.0
thetaN = 4.0
cWN = 7.5
beta = 0.5
cBN = 0.5
hBN = 3.0
deltaN = 0.05

# Cerfs
rD = 0.50
KD = 4.0
VstarD = 50.0
thetaD = 2.0
cWD = 2.0
hWD = 2.0
cBD = 0.3
hBD = 1.5
deltaD = 0.08

# Loups
epsilon1 = 0.25
epsilon2 = 0.5
eta = 0.01
mW = 0.15
muW = 0.10

# Ours
eB = 0.12
eV = 0.02
mB = 0.08



# CONDITIONS INITIALES


V0 = 440.0
N0 = 4.5
D0 = 2.5
W0 = 0.04
B0 = 0.25

u0 = np.array([V0, N0, D0, W0, B0])

t0 = 0.0
tf = 50.0
h = 0.05


# FONCTION DU SYSTÈME


def F(t, u):
    V, N, D, W, B = u

    kappaN = KN * V / (VstarN + V)
    kappaD = KD * V / (VstarD + V)

    phiW = cWN * N / (W + beta * N) + cWD * D / (hWD + D)

    dV = rV * V * (1.0 - V / KV) - alphaN * V * N / (nuN + V) - alphaD * V * D / (nuD + V) - alphaB * V * B / (nuB + V)

    dN = rN * N * (1.0 - (N / kappaN) ** thetaN) - cWN * N * W / (W + beta * N) - cBN * N * B / (hBN + N) - deltaN * N

    dD = rD * D * (1.0 - (D / kappaD) ** thetaD) - cWD * D * W / (hWD + D) - cBD * D * B / (hBD + D) - deltaD * D

    dW = (epsilon1 * np.log(phiW + eta) - epsilon2) * W - mW * W - muW * (np.sin(np.pi * t) ** 2) * W

    dB = (eB * (cBN * N / (hBN + N) + cBD * D / (hBD + D)) + eV * alphaB * V / (nuB + V) - mB) * B

    return np.array([dV, dN, dD, dW, dB])



# EULER EXPLICITE


def euler_explicite(u0, t0, tf, h):
    n = int((tf - t0) / h)
    t = np.linspace(t0, tf, n + 1)

    U = np.zeros((n + 1, 5))
    U[0] = u0

    for i in range(n):
        U[i + 1] = U[i] + h * F(t[i], U[i])

    return t, U



# RK4


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



# FACTORISATION LU


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



# JACOBIENNE NUMÉRIQUE


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



# NEWTON


def newton(G, x0, tol=1e-9, max_iter=20):
    x = x0.copy()

    for k in range(max_iter):
        gx = G(x)


        J = jacobienne_numerique(G, x)
        delta = resolution_lu(J, -gx)

        x = x + delta

    return x



# EULER IMPLICITE + NEWTON + LU


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



# EULER IMPLICITE + POINT FIXE


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

dossier = "figures"
os.makedirs(dossier, exist_ok=True)
if __name__ == "__main__":

    dossier = "figures"
    os.makedirs(dossier, exist_ok=True)

    
    # 1) Comparaison des méthodes sur le scénario de base
    
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

    # Comparaison méthodes : V
    plt.figure(figsize=(8, 4))
    plt.plot(t_euler, U_euler[:, 0], label="Euler explicite")
    plt.plot(t_rk4, U_rk4[:, 0], label="RK4")
    plt.plot(t_pf, U_pf[:, 0], label="Euler implicite point fixe")
    plt.plot(t_base, U_base[:, 0], label="Euler implicite Newton")
    plt.xlabel("Temps")
    plt.ylabel("V(t)")
    plt.title("Comparaison des méthodes : V(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "comparaison_methodes_V.png"), dpi=180)
    plt.close()

    # Comparaison méthodes : N
    plt.figure(figsize=(8, 4))
    plt.plot(t_euler, U_euler[:, 1], label="Euler explicite")
    plt.plot(t_rk4, U_rk4[:, 1], label="RK4")
    plt.plot(t_pf, U_pf[:, 1], label="Euler implicite point fixe")
    plt.plot(t_base, U_base[:, 1], label="Euler implicite Newton")
    plt.xlabel("Temps")
    plt.ylabel("N(t)")
    plt.title("Comparaison des méthodes : N(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "comparaison_methodes_N.png"), dpi=180)
    plt.close()

    # Comparaison méthodes : D
    plt.figure(figsize=(8, 4))
    plt.plot(t_euler, U_euler[:, 2], label="Euler explicite")
    plt.plot(t_rk4, U_rk4[:, 2], label="RK4")
    plt.plot(t_pf, U_pf[:, 2], label="Euler implicite point fixe")
    plt.plot(t_base, U_base[:, 2], label="Euler implicite Newton")
    plt.xlabel("Temps")
    plt.ylabel("D(t)")
    plt.title("Comparaison des méthodes : D(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "comparaison_methodes_D.png"), dpi=180)
    plt.close()

    # Comparaison méthodes : W
    plt.figure(figsize=(8, 4))
    plt.plot(t_euler, U_euler[:, 3], label="Euler explicite")
    plt.plot(t_rk4, U_rk4[:, 3], label="RK4")
    plt.plot(t_pf, U_pf[:, 3], label="Euler implicite point fixe")
    plt.plot(t_base, U_base[:, 3], label="Euler implicite Newton")
    plt.xlabel("Temps")
    plt.ylabel("W(t)")
    plt.title("Comparaison des méthodes : W(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "comparaison_methodes_W.png"), dpi=180)
    plt.close()

    # Comparaison méthodes : B
    plt.figure(figsize=(8, 4))
    plt.plot(t_euler, U_euler[:, 4], label="Euler explicite")
    plt.plot(t_rk4, U_rk4[:, 4], label="RK4")
    plt.plot(t_pf, U_pf[:, 4], label="Euler implicite point fixe")
    plt.plot(t_base, U_base[:, 4], label="Euler implicite Newton")
    plt.xlabel("Temps")
    plt.ylabel("B(t)")
    plt.title("Comparaison des méthodes : B(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "comparaison_methodes_B.png"), dpi=180)
    plt.close()

   
    # 2) Influence du pas de temps sur W
    

    # Grands pas : divergence
    t_h2, U_h2 = euler_implicite_newton(u0, t0, tf, 2.0)
    t_h1, U_h1 = euler_implicite_newton(u0, t0, tf, 1.0)
    t_h05, U_h05 = euler_implicite_newton(u0, t0, tf, 0.5)

    plt.figure(figsize=(8, 4))
    plt.plot(t_h2, U_h2[:, 3], label="h = 2")
    plt.plot(t_h1, U_h1[:, 3], label="h = 1")
    plt.plot(t_h05, U_h05[:, 3], label="h = 0.5")
    plt.xlabel("Temps")
    plt.ylabel("W(t)")
    plt.title("Divergence avec des grands pas de temps sur W(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "pas_temps_W_divergence.png"), dpi=180)
    plt.close()

    # Petits pas : convergence
    t_h01, U_h01 = euler_implicite_newton(u0, t0, tf, 0.1)
    t_h005, U_h005 = euler_implicite_newton(u0, t0, tf, 0.05)
    t_h001, U_h001 = euler_implicite_newton(u0, t0, tf, 0.01)
    t_h0005, U_h0005 = euler_implicite_newton(u0, t0, tf, 0.005)

    plt.figure(figsize=(8, 4))
    plt.plot(t_h01, U_h01[:, 3], label="h = 0.1")
    plt.plot(t_h005, U_h005[:, 3], label="h = 0.05")
    plt.plot(t_h001, U_h001[:, 3], label="h = 0.01")
    plt.plot(t_h0005, U_h0005[:, 3], label="h = 0.005")
    plt.xlabel("Temps")
    plt.ylabel("W(t)")
    plt.title("Convergence avec des petits pas de temps sur W(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "pas_temps_W_convergence.png"), dpi=180)
    plt.close()

    
    # 3) Scénario de base
    

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

    
    # 4) Effet du prélèvement humain muW
    

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

    # Effet prélèvement : V
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

    # Effet prélèvement : N
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

    # Effet prélèvement : D
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

    # Effet prélèvement : W
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

    # Effet prélèvement : B
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

    
    # 5) Scénario Yellowstone
    

    u0_yellowstone = np.array([V0, N0, D0, 0.0001, B0])

    t1, U1 = euler_implicite_newton(u0_yellowstone, 0.0, 10.0, h)

    u_reintro = U1[-1].copy()
    u_reintro[3] = 0.04

    t2, U2 = euler_implicite_newton(u_reintro, 10.0, 50.0, h)

    t_y = np.concatenate((t1, t2[1:]))
    U_y = np.vstack((U1, U2[1:]))

    # Yellowstone : V
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

    # Yellowstone : N
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

    # Yellowstone : D
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

    # Yellowstone : W
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

    # Yellowstone : B
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

    
    # 6) Affichage console
    

    print("-" * 70)
    print("Temps de calcul")
    print("Euler explicite :", round(temps_euler, 4), "s")
    print("RK4 :", round(temps_rk4, 4), "s")
    print("Euler implicite point fixe :", round(temps_pf, 4), "s")
    print("Euler implicite Newton + LU :", round(temps_newton, 4), "s")
    print("-" * 70)
