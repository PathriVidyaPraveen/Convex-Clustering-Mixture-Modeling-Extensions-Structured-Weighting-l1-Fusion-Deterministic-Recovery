# Code developed : Abhinav

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ──────────────────────────────────────────────
# 0.  Hyper-parameters
# ──────────────────────────────────────────────
SEED        = 42
N           = 500
D           = 2
K_TRUE      = 3

LAMBDA      = 0.05
RHO         = 1.0
MAX_ITER    = 300
TOL         = 1e-6

CLUSTER_EPS = 1e-3
USE_SPARSE  = True

# ──────────────────────────────────────────────
# 1.  Synthetic data
# ──────────────────────────────────────────────
rng = np.random.default_rng(SEED)

n_per = N // K_TRUE
counts = [n_per, n_per, N - 2 * n_per]

means = np.array([
    [0.0, 0.0],
    [1.8, 0.5],
    [0.9, 1.6],
])

covs = [
    np.array([[0.25, 0.18], [0.18, 0.15]]),
    np.array([[0.20, -0.12], [-0.12, 0.10]]),
    np.array([[0.10, 0.0],  [0.0,  0.30]]),
]

A_list, labels_list = [], []
for k in range(K_TRUE):
    pts = rng.multivariate_normal(means[k], covs[k], counts[k])
    A_list.append(pts)
    labels_list.append(np.full(counts[k], k, dtype=int))

A      = np.vstack(A_list)
labels = np.concatenate(labels_list)

# ──────────────────────────────────────────────
# 2.  Weight matrix  W  (inverse-distance, O(N^2))   # <-- changed heading
# ──────────────────────────────────────────────
# Inverse-distance weights: w_ij = 1 / ||a_i - a_j||  # <-- changed heading

gamma = N ** (3.0 / (4.0 * D))

diff_ij = A[:, None, :] - A[None, :, :]
dist_ij = np.linalg.norm(diff_ij, axis=-1)

W = 1.0 / (dist_ij+1e-12)

if USE_SPARSE:
    omega = (D + 4.0 / 3.0) * (np.log(gamma) / gamma)
    W[dist_ij > omega] = 0.0

np.fill_diagonal(W, 0.0)

# ──────────────────────────────────────────────
# 3.  ADMM solver
# ──────────────────────────────────────────────
X = A.copy()
Z = np.zeros((N, N, D))
U = np.zeros((N, N, D))

lam_w_over_rho = (LAMBDA * W) / RHO

print(f"SON-ADMM  |  N={N}, D={D}, K={K_TRUE}")
print(f"gamma={gamma:.4f}, lambda={LAMBDA}, rho={RHO}, "
      f"sparse={USE_SPARSE}" + (f", omega={omega:.4f}" if USE_SPARSE else ""))
print(f"Non-zero weights: {np.sum(W > 0)} / {N*N}")
print("-" * 55)

for iteration in range(1, MAX_ITER + 1):

    X_old = X.copy()

    V = X[:, None, :] - X[None, :, :] + U
    V_norm = np.linalg.norm(V, axis=-1, keepdims=True)
    V_norm = np.where(V_norm == 0, 1.0, V_norm)

    shrink = np.maximum(
        1.0 - lam_w_over_rho[:, :, None] / V_norm,
        0.0
    )

    Z = shrink * V
    Z = 0.5 * (Z - Z.transpose(1, 0, 2))

    sum_x = X.sum(axis=0, keepdims=True)
    sum_ZmU = (Z - U).sum(axis=1)

    X = (A + RHO * (sum_x + sum_ZmU)) / (1.0 + RHO * N)

    U = U + X[:, None, :] - X[None, :, :] - Z

    primal_res = np.linalg.norm(X - X_old)
    dual_res   = RHO * np.linalg.norm(
        (X[:, None, :] - X[None, :, :]) - (X_old[:, None, :] - X_old[None, :, :])
    )

    if iteration % 50 == 0 or iteration == 1:
        print(f"  iter {iteration:4d} | primal={primal_res:.2e} | dual={dual_res:.2e}")

    if primal_res < TOL and dual_res < TOL and iteration > 10:
        print(f"  Converged at iteration {iteration}.")
        break

X_star = X

# ──────────────────────────────────────────────
# 4.  Cluster extraction
# ──────────────────────────────────────────────
from scipy.spatial.distance import cdist

dist_x = cdist(X_star, X_star)
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
pred_labels  = np.searchsorted(unique_roots, cluster_ids)

n_clusters_found = len(unique_roots)

# ──────────────────────────────────────────────
# 5.  Evaluation
# ──────────────────────────────────────────────
from collections import Counter

pred_to_true = {}
for pid in np.unique(pred_labels):
    mask = pred_labels == pid
    majority = Counter(labels[mask]).most_common(1)[0][0]
    pred_to_true[pid] = majority

correctly_assigned = sum(
    pred_to_true[pred_labels[i]] == labels[i] for i in range(N)
)
s1 = correctly_assigned / N

mapped_true_labels = set(pred_to_true.values())
s3 = len(mapped_true_labels)

print()
print("=" * 55)
print(f"  Core Recovery  s1 = {correctly_assigned}/{N} = {s1:.4f}")
print(f"  Distinct Clusters found: {n_clusters_found}")
print(f"  Distinct Clusters s3 = {s3}/{K_TRUE}")
print("=" * 55)

# ──────────────────────────────────────────────
# 6.  Visualisation
# ──────────────────────────────────────────────
COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261', '#264653', '#E9C46A']
GT_COLORS  = [COLORS[k] for k in labels]
REC_COLORS = [COLORS[pred_labels[i] % len(COLORS)] for i in range(N)]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    "Localized Sum-of-Norms Clustering via ADMM\n"
    r"$w_{ij} = 1 / \|a_i - a_j\|$",   # <-- ONLY change here
    fontsize=13, fontweight='bold'
)

# Panel 1
ax = axes[0]
for k in range(K_TRUE):
    mask = labels == k
    ax.scatter(A[mask, 0], A[mask, 1],
               c=COLORS[k], s=14, alpha=0.7, linewidths=0,
               label=f"Cluster {k+1}")
ax.set_title(f"Observed data $a_i$  (ground truth, N={N})")
ax.legend()

# Panel 2 (THIS HAS STARS + CLUSTER COUNT — untouched)
ax = axes[1]
for pid in np.unique(pred_labels):
    mask = pred_labels == pid
    col  = COLORS[pid % len(COLORS)]
    ax.scatter(X_star[mask, 0], X_star[mask, 1],
               c=col, s=14, alpha=0.85, linewidths=0)

unique_reps = np.array([
    X_star[pred_labels == pid].mean(axis=0)
    for pid in np.unique(pred_labels)
])

ax.scatter(unique_reps[:, 0], unique_reps[:, 1],
           c='black', marker='*', s=180, zorder=5,
           label=f"Representatives ({n_clusters_found} clusters)")

ax.set_title(
    f"Converged representatives $x^*_i$ "
    f"(s1={s1:.3f}, s3={s3}/{K_TRUE})"
)
ax.legend()

plt.tight_layout()
out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "son_admm_result.png")
plt.savefig(out_path)
plt.show()