# Code developed by : Gagan + Prajin

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
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, required=True)
parser.add_argument("--lambda_", type=float, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--silent", action="store_true")
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

SIGMA = args.sigma
LAMBDA = args.lambda_
SEED = args.seed
# ──────────────────────────────────────────────
# 0.  Hyper-parameters
# ──────────────────────────────────────────────
SEED        = 42
N           = 500          # number of data points
D           = 2            # ambient dimension
K_TRUE      = 3            # number of ground-truth clusters

#LAMBDA      = 0.05         # regularisation strength  (tune if needed)
RHO         = 1.0          # ADMM penalty parameter
MAX_ITER    = 300          # maximum ADMM iterations
TOL         = 1e-6         # primal/dual residual tolerance

CLUSTER_EPS = 1e-3         # grouping threshold for recovered clusters
USE_SPARSE  = True         # enable weight truncation for efficiency

# ──────────────────────────────────────────────
# 1.  Synthetic data  (3 anisotropic Gaussians
#     with overlapping convex hulls)
# ──────────────────────────────────────────────
rng = np.random.default_rng(SEED)

n_per = N // K_TRUE
counts = [n_per, n_per, N - 2 * n_per]

# Cluster centres — placed so convex hulls overlap
means = np.array([
    [ 0.0,  0.0],
    [ 1.8,  0.5],
    [ 0.9,  1.6],
])

# Anisotropic covariances
covs = [
    np.array([[0.25, 0.18], [0.18, 0.15]]),   # cluster 0 – elongated NE
    np.array([[0.20, -0.12], [-0.12, 0.10]]),  # cluster 1 – elongated NW
    np.array([[0.10, 0.0],  [0.0,  0.30]]),    # cluster 2 – elongated N
]

A_list, labels_list = [], []
for k in range(K_TRUE):
    pts = rng.multivariate_normal(means[k], covs[k], counts[k])
    A_list.append(pts)
    labels_list.append(np.full(counts[k], k, dtype=int))

A      = np.vstack(A_list)          # (N, D)  observed data
labels = np.concatenate(labels_list) # (N,)    ground-truth labels

# ──────────────────────────────────────────────
# 2.  Weight matrix  W  (pre-computed, O(N^2))
# ──────────────────────────────────────────────
diff_ij = A[:, None, :] - A[None, :, :]          # (N, N, D)
dist_sq_ij = np.sum(diff_ij**2, axis=-1)         # (N, N)
#SIGMA = 1.0   # tune this

W = np.exp(-dist_sq_ij / (2 * SIGMA**2))
if USE_SPARSE:
    threshold = 1e-3
    W[W < threshold] = 0.0
np.fill_diagonal(W, 0.0)
# ──────────────────────────────────────────────
# 3.  ADMM solver
# ──────────────────────────────────────────────
# Variables:
#   X  (N, D)        – primal cluster representatives
#   Z  (N, N, D)     – auxiliary pairwise differences
#   U  (N, N, D)     – scaled dual variables (U_ij)
#
# We exploit antisymmetry Z_ij = -Z_ji to halve memory by
# working with the full (N,N,D) arrays but enforcing antisymmetry
# after each Z-update.

X = A.copy()                              # warm start: x_i = a_i
Z = np.zeros((N, N, D), dtype=float)
U = np.zeros((N, N, D), dtype=float)

# Pre-compute lambda*W / rho  shape (N, N) for use in soft-threshold
lam_w_over_rho = (LAMBDA * W) / RHO      # (N, N)

#print(f"SON-ADMM  |  N={N}, D={D}, K={K_TRUE}")
#print(f"Gaussian kernel | sigma={SIGMA}, lambda={LAMBDA}, rho={RHO}, "
      #f"sparse={USE_SPARSE}")
#print(f"Non-zero weights: {np.sum(W > 0)} / {N*N}")
#print("-" * 55)

for iteration in range(1, MAX_ITER + 1):

    X_old = X.copy()

    # ── Z-update (vectorised soft-thresholding) ──────────────
    # V_ij = x_i - x_j + U_ij  shape (N, N, D)
    V = X[:, None, :] - X[None, :, :] + U          # (N, N, D)

    V_norm = np.linalg.norm(V, axis=-1, keepdims=True)  # (N, N, 1)
    V_norm = np.where(V_norm == 0, 1.0, V_norm)         # avoid /0

    # Shrinkage factor: max(1 - lam*w_ij / (rho*||V_ij||), 0)
    shrink = np.maximum(
        1.0 - lam_w_over_rho[:, :, None] / V_norm,
        0.0
    )                                                # (N, N, 1)

    Z = shrink * V                                   # (N, N, D)

    # Enforce antisymmetry: Z_ij = (Z_ij - Z_ji) / 2
    Z = 0.5 * (Z - Z.transpose(1, 0, 2))

    # ── X-update (closed-form primal) ─────────────────────────
    # x_i = (a_i + rho * sum_j (x_j + Z_ij - U_ij)) / (1 + rho*n)
    #
    # sum_j (x_j + Z_ij - U_ij) for each i:
    #   = sum_j x_j  +  sum_j Z_ij  -  sum_j U_ij
    sum_x = X.sum(axis=0, keepdims=True)             # (1, D)
    sum_ZmU = (Z - U).sum(axis=1)                    # (N, D)  sum over j

    X = (A + RHO * (sum_x + sum_ZmU)) / (1.0 + RHO * N)

    # ── U-update (dual ascent) ────────────────────────────────
    U = U + X[:, None, :] - X[None, :, :] - Z

    # ── Convergence check ─────────────────────────────────────
    primal_res = np.linalg.norm(X - X_old)
    dual_res   = RHO * np.linalg.norm(
        (X[:, None, :] - X[None, :, :]) - (X_old[:, None, :] - X_old[None, :, :])
    )

    if iteration % 50 == 0 or iteration == 1:
        #print(f"  iter {iteration:4d} | primal={primal_res:.2e} | dual={dual_res:.2e}")
        continue

    if primal_res < TOL and dual_res < TOL and iteration > 10:
        #print(f"  Converged at iteration {iteration}.")
        break

X_star = X   # converged cluster representatives

# ──────────────────────────────────────────────
# 4.  Cluster extraction
#     Group points with ||x_i* - x_j*|| < CLUSTER_EPS
# ──────────────────────────────────────────────
from scipy.spatial.distance import cdist  # lightweight, always available

dist_x = cdist(X_star, X_star)           # (N, N)
# Union-Find to label connected components
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

rows, cols = np.where((dist_x < CLUSTER_EPS) & (dist_x > 0))
for r, c in zip(rows, cols):
    union(r, c)

cluster_ids = np.array([find(i) for i in range(N)])
unique_roots = np.unique(cluster_ids)
pred_labels  = np.searchsorted(unique_roots, cluster_ids)   # 0-indexed

n_clusters_found = len(unique_roots)

# ──────────────────────────────────────────────
# 5.  Evaluation metrics  (s1, s3)
# ──────────────────────────────────────────────
# Core sets V_m: all points of ground-truth cluster m
# (the full cluster is the "core" for synthetic data)
# s1 = fraction of core points correctly assigned (majority vote per cluster)
# s3 = number of distinct recovered clusters that map to distinct true clusters

from collections import Counter

# For each predicted cluster, find the majority ground-truth label
pred_to_true = {}
for pid in np.unique(pred_labels):
    mask = pred_labels == pid
    majority = Counter(labels[mask]).most_common(1)[0][0]
    pred_to_true[pid] = majority

# s1: core recovery
correctly_assigned = sum(
    pred_to_true[pred_labels[i]] == labels[i] for i in range(N)
)
s1 = correctly_assigned / N

# s3: number of distinct recovered clusters that map to distinct true labels
mapped_true_labels = set(pred_to_true.values())
s3 = len(mapped_true_labels)

#print()
#print("=" * 55)
#print(f"  Core Recovery  s1 = {correctly_assigned}/{N} = {s1:.4f}")
#print(f"  Distinct Clusters found: {n_clusters_found}")
#print(f"  Distinct Clusters s3 = {s3}/{K_TRUE}")
#print("=" * 55)
print(f"{SIGMA},{LAMBDA},{n_clusters_found},{s1},{s3}")
# ──────────────────────────────────────────────
# 6.  Visualisation  (two-panel plot)
# ──────────────────────────────────────────────
if args.plot:
    COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261', '#264653', '#E9C46A']
    GT_COLORS  = [COLORS[k] for k in labels]
    REC_COLORS = [COLORS[pred_labels[i] % len(COLORS)] for i in range(N)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Gaussian Kernel Sum-of-Norms Clustering via ADMM\n"
        rf"$w_{{ij}}=\exp\left(-\frac{{\|a_i-a_j\|^2}}{{2\sigma^2}}\right)$, "
        rf"$\sigma={SIGMA}$",
        fontsize=13, fontweight='bold'
    )

    # ── Panel 1: observed data with ground-truth labels ──────────
    ax = axes[0]
    for k in range(K_TRUE):
        mask = labels == k
        ax.scatter(A[mask, 0], A[mask, 1],
                   c=COLORS[k], s=14, alpha=0.7, linewidths=0,
                   label=f"Cluster {k+1}")
    ax.set_title(f"Observed data $a_i$  (ground truth, N={N})", fontsize=11)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.legend(fontsize=9, markerscale=1.5)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle='--', alpha=0.3)

    # ── Panel 2: converged representatives x* ────────────────────
    ax = axes[1]
    for pid in np.unique(pred_labels):
        mask = pred_labels == pid
        col  = COLORS[pid % len(COLORS)]
        ax.scatter(X_star[mask, 0], X_star[mask, 1],
                   c=col, s=14, alpha=0.85, linewidths=0)

    # Mark the unique representative positions
    unique_reps = np.array([X_star[pred_labels == pid].mean(axis=0)
                            for pid in np.unique(pred_labels)])
    ax.scatter(unique_reps[:, 0], unique_reps[:, 1],
               c='black', marker='*', s=180, zorder=5,
               label=f"Representatives ({n_clusters_found} clusters)")

    ax.set_title(
        f"Converged representatives $x^*_i$  "
        f"(s1={s1:.3f}, s3={s3}/{K_TRUE})",
        fontsize=11
    )
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.legend(fontsize=9)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
out_dir = "outputs-gagan"
out_path = os.path.join(out_dir, "son_admm_result.png")

#print(f"\nPlot saved → {out_path}")

