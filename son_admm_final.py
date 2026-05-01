# Code developed by : Praveen

"""
Localized Sum-of-Norms (SON) Clustering via ADMM
=================================================
Theoretically grounded in:
  Dunlap & Mourrat, "Local Versions of Sum-of-Norms Clustering" (arXiv:2109.09589v3)

Objective:
    f(x) = (1/2) * sum_i ||x_i - a_i||^2  +  lambda * sum_{i<j} w_ij * ||x_i - x_j||_2

Weight function (localized kernel):
    w_ij = gamma^(d+1) * exp(-gamma * ||a_i - a_j||_2)
    gamma = N^(3/(4d))   [scaling law from Theorem 1.2]

Optional sparsity truncation:
    w_ij = 0  if  ||a_i - a_j||_2 > (d + 4/3) * gamma^{-1} * log(gamma)

ADMM updates:
    V_ij   = x_i - x_j + U_ij
    Z_ij   = max(1 - lambda*w_ij / (rho*||V_ij||), 0) * V_ij   [soft-threshold]
    x_i    = (a_i + rho * sum_j(x_j + Z_ij - U_ij)) / (1 + rho*n)
    U_ij   = U_ij + x_i - x_j - Z_ij

Plots produced
--------------
  Figure 1 – Cluster Path vs Lambda  (clusterpath / tree-type plot)
             Number-of-clusters curve + trajectory lines showing how
             representative points agglomerate as lambda grows.
  Figure 2 – Two-panel result plot
             Panel A: observed data a_i with ground-truth colours
             Panel B: converged representatives x*_i at chosen lambda
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from collections import Counter

# ═══════════════════════════════════════════════════════════════
# 0.  Global hyper-parameters
# ═══════════════════════════════════════════════════════════════
SEED          = 42
N_MAIN        = 500       # points for the final result plot
N_PATH        = 120       # smaller set for the clusterpath sweep (speed)
D             = 2         # ambient dimension
K_TRUE        = 3         # ground-truth clusters

LAMBDA_MAIN   = 0.05      # lambda for the two-panel result
RHO           = 1.0       # ADMM penalty parameter
MAX_ITER_MAIN = 300       # iterations for the main run
MAX_ITER_PATH = 60        # iterations per lambda in the sweep
TOL_MAIN      = 1e-6
TOL_PATH      = 1e-3      # looser for speed

CLUSTER_EPS   = 1e-3      # grouping threshold
USE_SPARSE    = True      # weight truncation

# lambda grid: 20 values log-spaced
LAMBDA_GRID   = np.logspace(-3, 0, 20)

# ═══════════════════════════════════════════════════════════════
# 1.  Data generation
# ═══════════════════════════════════════════════════════════════
def generate_data(N, seed=SEED):
    rng    = np.random.default_rng(seed)
    n_per  = N // K_TRUE
    counts = [n_per, n_per, N - 2 * n_per]
    means  = np.array([[0.0, 0.0], [1.8, 0.5], [0.9, 1.6]])
    covs   = [
        np.array([[0.25,  0.18], [ 0.18, 0.15]]),
        np.array([[0.20, -0.12], [-0.12, 0.10]]),
        np.array([[0.10,  0.00], [ 0.00, 0.30]]),
    ]
    A_list, L_list = [], []
    for k in range(K_TRUE):
        A_list.append(rng.multivariate_normal(means[k], covs[k], counts[k]))
        L_list.append(np.full(counts[k], k, dtype=int))
    return np.vstack(A_list), np.concatenate(L_list)

# ═══════════════════════════════════════════════════════════════
# 2.  Weight matrix
# ═══════════════════════════════════════════════════════════════
def build_weights(A, D, use_sparse=USE_SPARSE):
    N     = len(A)
    gamma = N ** (3.0 / (4.0 * D))
    dist  = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=-1)
    W     = (gamma ** (D + 1)) * np.exp(-gamma * dist)
    if use_sparse:
        omega = (D + 4.0 / 3.0) * (np.log(gamma) / gamma)
        W[dist > omega] = 0.0
    np.fill_diagonal(W, 0.0)
    return W, gamma

# ═══════════════════════════════════════════════════════════════
# 3.  ADMM solver
# ═══════════════════════════════════════════════════════════════
def run_admm(A, W, lam, rho, max_iter, tol, X_init=None, verbose=False):
    N, D = A.shape
    X    = A.copy() if X_init is None else X_init.copy()
    Z    = np.zeros((N, N, D))
    U    = np.zeros((N, N, D))
    lwr  = (lam * W) / rho

    for it in range(1, max_iter + 1):
        X_old = X.copy()

        V      = X[:, None, :] - X[None, :, :] + U
        V_norm = np.linalg.norm(V, axis=-1, keepdims=True)
        V_norm = np.where(V_norm == 0, 1.0, V_norm)
        Z      = np.maximum(1.0 - lwr[:, :, None] / V_norm, 0.0) * V
        Z      = 0.5 * (Z - Z.transpose(1, 0, 2))

        sum_x   = X.sum(axis=0, keepdims=True)
        sum_ZmU = (Z - U).sum(axis=1)
        X = (A + rho * (sum_x + sum_ZmU)) / (1.0 + rho * N)

        U = U + X[:, None, :] - X[None, :, :] - Z

        p_res = np.linalg.norm(X - X_old)
        d_res = rho * np.linalg.norm(
            (X[:, None, :] - X[None, :, :]) -
            (X_old[:, None, :] - X_old[None, :, :])
        )
        if verbose and (it % 50 == 0 or it == 1):
            print(f"  iter {it:4d}  |  primal={p_res:.2e}  dual={d_res:.2e}")
        if p_res < tol and d_res < tol and it > 10:
            if verbose:
                print(f"  Converged at iteration {it}.")
            break
    return X

# ═══════════════════════════════════════════════════════════════
# 4.  Union-Find cluster extraction
# ═══════════════════════════════════════════════════════════════
def extract_clusters(X_star, eps=CLUSTER_EPS):
    N      = len(X_star)
    parent = np.arange(N)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv

    dist_x = cdist(X_star, X_star)
    r, c   = np.where((dist_x < eps) & (dist_x > 0))
    for ri, ci in zip(r, c):
        union(ri, ci)

    cids  = np.array([find(i) for i in range(N)])
    roots = np.unique(cids)
    pred  = np.searchsorted(roots, cids)
    return pred, len(roots)

# ═══════════════════════════════════════════════════════════════
# 5.  Build datasets & weight matrices
# ═══════════════════════════════════════════════════════════════
A_main, labels_main = generate_data(N_MAIN)
A_path, labels_path = generate_data(N_PATH)

W_main, gamma_main  = build_weights(A_main, D)
W_path, gamma_path  = build_weights(A_path, D)

print(f"SON-ADMM  |  N={N_MAIN} (main)  N_path={N_PATH} (sweep)  D={D}  K={K_TRUE}")
print(f"gamma_main={gamma_main:.4f}  gamma_path={gamma_path:.4f}")
print(f"Non-zero weights (main): {int(np.sum(W_main > 0))} / {N_MAIN**2}")
print("-" * 60)

# ═══════════════════════════════════════════════════════════════
# 6.  Clusterpath sweep  (small dataset, warm-started)
# ═══════════════════════════════════════════════════════════════
print("\nSweeping lambda grid for clusterpath …")
path_n = []
path_X = []
X_warm = None

for lam in LAMBDA_GRID:
    Xs = run_admm(A_path, W_path, lam, RHO,
                  MAX_ITER_PATH, TOL_PATH, X_init=X_warm)
    _, n_c = extract_clusters(Xs)
    path_n.append(n_c)
    path_X.append(Xs.copy())
    X_warm = Xs.copy()
    print(f"  lambda={lam:.5f}  ->  {n_c:3d} clusters")

path_X = np.array(path_X)   # (n_lam, N_PATH, D)
print("Sweep done.\n")

# ═══════════════════════════════════════════════════════════════
# 7.  Main ADMM run  (full dataset, LAMBDA_MAIN)
# ═══════════════════════════════════════════════════════════════
print(f"Main run: N={N_MAIN}, lambda={LAMBDA_MAIN} …")
X_star = run_admm(A_main, W_main, LAMBDA_MAIN, RHO,
                  MAX_ITER_MAIN, TOL_MAIN, verbose=True)

pred_labels, n_found = extract_clusters(X_star)

pred_to_true = {}
for pid in np.unique(pred_labels):
    mask = pred_labels == pid
    pred_to_true[pid] = Counter(labels_main[mask]).most_common(1)[0][0]

correct = sum(pred_to_true[pred_labels[i]] == labels_main[i]
              for i in range(N_MAIN))
s1 = correct / N_MAIN
s3 = len(set(pred_to_true.values()))

print()
print("=" * 60)
print(f"  Core Recovery       s1 = {correct}/{N_MAIN} = {s1:.4f}")
print(f"  Distinct groups found   : {n_found}")
print(f"  Component Distinctness s3 = {s3}/{K_TRUE}")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 8.  FIGURE 1 – Clusterpath / Tree-type plot
#
#     Replicates Figure 1 style from the project report:
#     "Cluster Path vs Lambda" — step curve showing agglomeration.
#     PLUS the trajectory panel (representative paths in PC-1).
# ═══════════════════════════════════════════════════════════════
COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261', '#264653', '#E9C46A']

# Project path onto first principal component of A_path
A_c  = A_path - A_path.mean(axis=0)
Vt   = np.linalg.svd(A_c, full_matrices=False)[2]
pc1  = Vt[0]                          # shape (D,)
traj = path_X @ pc1                   # (n_lam, N_PATH)  PC-1 projections

fig1, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(11, 8),
    gridspec_kw={'height_ratios': [3, 1.5]},
    sharex=True
)
fig1.suptitle(
    "Clusterpath  —  Cluster Path vs  λ\n"
    "Localized SON Clustering  |  Agglomeration via ADMM",
    fontsize=13, fontweight='bold'
)

# ── Top panel : individual + centroid trajectories ───────────
for pt in range(N_PATH):
    ax_top.plot(LAMBDA_GRID, traj[:, pt],
                color=COLORS[labels_path[pt]],
                alpha=0.18, linewidth=0.65, rasterized=True)

for k in range(K_TRUE):
    mk = labels_path == k
    ax_top.plot(LAMBDA_GRID, traj[:, mk].mean(axis=1),
                color=COLORS[k], linewidth=3.0, alpha=0.95,
                label=f"Centroid – cluster {k + 1}", zorder=5)

ax_top.axvline(LAMBDA_MAIN, color='black', linestyle=':', linewidth=1.8,
               label=f'λ_main = {LAMBDA_MAIN}')
ax_top.axvspan(LAMBDA_MAIN * 0.4, LAMBDA_MAIN * 2.5,
               alpha=0.07, color='gray')
ax_top.set_xscale('log')
ax_top.set_ylabel(r"Projection onto PC-1  $(x_i^* \cdot \hat{v}_1)$", fontsize=10)
ax_top.set_title(
    r"Representative trajectories $x_i^*(\lambda)$ — "
    r"points fuse as $\lambda$ increases",
    fontsize=10
)
ax_top.legend(fontsize=9, loc='upper left', framealpha=0.88)
ax_top.grid(True, which='both', linestyle='--', alpha=0.25)

# ── Bottom panel : step curve of #clusters ───────────────────
ax_bot.step(LAMBDA_GRID, path_n, where='post',
            color='#264653', linewidth=2.2, zorder=4, label='# clusters')
ax_bot.scatter(LAMBDA_GRID, path_n,
               color='#E9C46A', edgecolors='#264653',
               s=45, zorder=5, linewidths=0.9)
ax_bot.fill_between(LAMBDA_GRID, path_n, step='post',
                    alpha=0.18, color='#457B9D')
ax_bot.axvline(LAMBDA_MAIN, color='black', linestyle=':', linewidth=1.8)
ax_bot.axhline(K_TRUE, color='#E63946', linestyle='--', linewidth=1.5,
               label=f'True K = {K_TRUE}')

# Annotate operating point
idx_op  = np.argmin(np.abs(LAMBDA_GRID - LAMBDA_MAIN))
k_op    = path_n[idx_op]
y_range = max(path_n) - min(path_n) + 1
ax_bot.annotate(
    f"λ={LAMBDA_MAIN}\nK={k_op}",
    xy=(LAMBDA_MAIN, k_op),
    xytext=(LAMBDA_MAIN * 5, k_op + y_range * 0.15),
    arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
    fontsize=9, color='black',
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85)
)
ax_bot.set_xlabel("Regularisation parameter  λ  (log scale)", fontsize=10)
ax_bot.set_ylabel("Number of clusters", fontsize=10)
ax_bot.set_title("Number of distinct clusters vs λ", fontsize=10)
ax_bot.legend(fontsize=9, loc='upper right', framealpha=0.88)
ax_bot.grid(True, which='both', linestyle='--', alpha=0.25)

plt.tight_layout()
path1 = "/mnt/user-data/outputs/son_clusterpath.png"
fig1.savefig(path1, dpi=150, bbox_inches='tight')
print(f"Clusterpath figure → {path1}")

# ═══════════════════════════════════════════════════════════════
# 9.  FIGURE 2 – Two-panel result at LAMBDA_MAIN
# ═══════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig2.suptitle(
    "Localized Sum-of-Norms Clustering via ADMM\n"
    r"$w_{ij}=\gamma^{d+1}e^{-\gamma\|a_i-a_j\|}$,  "
    rf"$\gamma = N^{{3/(4d)}} = {gamma_main:.2f}$,  "
    rf"$\lambda = {LAMBDA_MAIN}$",
    fontsize=13, fontweight='bold'
)

# Panel A – ground truth
ax = axes[0]
for k in range(K_TRUE):
    mask = labels_main == k
    ax.scatter(A_main[mask, 0], A_main[mask, 1],
               c=COLORS[k], s=14, alpha=0.7, linewidths=0,
               label=f"Cluster {k + 1}")
ax.set_title(f"Observed data  $a_i$  (ground truth,  N={N_MAIN})", fontsize=11)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.legend(fontsize=9, markerscale=1.5)
ax.set_aspect('equal', adjustable='datalim')
ax.grid(True, linestyle='--', alpha=0.3)

# Panel B – recovered representatives
ax = axes[1]
for pid in np.unique(pred_labels):
    mask = pred_labels == pid
    ax.scatter(X_star[mask, 0], X_star[mask, 1],
               c=COLORS[pid % len(COLORS)], s=14, alpha=0.85, linewidths=0)

unique_reps = np.array([X_star[pred_labels == pid].mean(axis=0)
                         for pid in np.unique(pred_labels)])
ax.scatter(unique_reps[:, 0], unique_reps[:, 1],
           c='black', marker='*', s=200, zorder=5,
           label=f"Representatives  ({n_found} groups)")

ax.set_title(
    f"Converged representatives  $x^*_i$\n"
    f"s1 = {correct}/{N_MAIN} = {s1:.3f}   |   s3 = {s3}/{K_TRUE}",
    fontsize=10
)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.legend(fontsize=9)
ax.set_aspect('equal', adjustable='datalim')
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
path2 = "/mnt/user-data/outputs/son_admm_result.png"
fig2.savefig(path2, dpi=150, bbox_inches='tight')
print(f"Result figure        → {path2}")

plt.show()
print("\nAll done.")
