# Code developed by : Abhinav + Gagan

"""
L1 Sum-of-Norms Clustering Simulation
======================================
Replicates the computational experiments from:
  "Recovery of a Mixture of Gaussians by Sum-of-norms Clustering"
  Jiang, Vavasis & Zhai (JMLR 2020)

Modified to use the L1 norm in both the fidelity and fusion terms:

  min_{x_1,...,x_n in R^d}  sum_i ||x_i - a_i||_1
                            + lambda * sum_{i<j} ||x_i - x_j||_1

Key differences from the L2 (paper) version:
  - Uses Manhattan distance instead of Euclidean
  - L1 fusion promotes more aggressive coordinate-wise sparsity
  - Reformulable as a Linear Program (LP) — often faster than L2 ADMM
  - Cluster boundaries are "boxy" (L1 ball) rather than spherical
"""

import numpy as np
import cvxpy as cp
import itertools
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_mixture_of_gaussians(n, d, K, means, sigma, weights=None, seed=42):
    """
    Draw n i.i.d. samples from a K-component spherical Gaussian mixture.

    Parameters
    ----------
    n      : number of samples
    d      : ambient dimension
    K      : number of components
    means  : list/array of shape (K, d) — cluster means µ_m
    sigma  : scalar or array of length K — per-cluster std dev σ_m
    weights: mixture weights w_1,...,w_K (uniform if None)
    seed   : random seed for reproducibility

    Returns
    -------
    A      : (n, d) data matrix
    labels : (n,)  ground-truth component index for each point
    """
    rng = np.random.default_rng(seed)
    means = np.asarray(means, dtype=float)   # (K, d)
    if weights is None:
        weights = np.ones(K) / K
    weights = np.asarray(weights, dtype=float)
    sigma = np.broadcast_to(sigma, (K,))

    # Choose component indices
    labels = rng.choice(K, size=n, p=weights)
    A = np.zeros((n, d))
    for i, m in enumerate(labels):
        A[i] = rng.normal(loc=means[m], scale=sigma[m], size=d)

    return A, labels


def compute_Vm(A, labels, means, sigma, theta, K):
    """
    V_m = {i : ||a_i - µ_m|| <= theta * sigma_m}
    Returns a dict m -> set of indices in V_m (using true L2 distance to mean).
    """
    Vm = {}
    for m in range(K):
        dists = np.linalg.norm(A - means[m], axis=1)
        Vm[m] = set(np.where((dists <= theta * sigma[m]) & (labels == m))[0])
    return Vm


# ─────────────────────────────────────────────────────────────────────────────
# 2. L1 SUM-OF-NORMS CLUSTERING SOLVER
# ─────────────────────────────────────────────────────────────────────────────

def solve_l1_son_clustering(A, lam, solver=None, verbose=False):
    """
    Solve the L1 sum-of-norms clustering problem via CVXPY:

        min  sum_i ||x_i - a_i||_1  +  lam * sum_{i<j} ||x_i - x_j||_1

    Parameters
    ----------
    A      : (n, d) data matrix
    lam    : regularisation parameter lambda
    solver : CVXPY solver string (None = auto-select)
    verbose: print solver output

    Returns
    -------
    X_opt  : (n, d) optimal solution matrix
    """
    n, d = A.shape

    # Decision variables: one d-vector per data point
    X = cp.Variable((n, d))

    # Fidelity term: sum_i ||x_i - a_i||_1
    fidelity = cp.sum([cp.norm(X[i] - A[i], 1) for i in range(n)])

    # Fusion term: sum_{i<j} ||x_i - x_j||_1
    fusion = cp.sum([cp.norm(X[i] - X[j], 1)
                     for i, j in itertools.combinations(range(n), 2)])

    objective = cp.Minimize(fidelity + lam * fusion)
    prob = cp.Problem(objective)

    # Solver priority: MOSEK > GUROBI > CLARABEL > ECOS
    solver_order = [cp.MOSEK, cp.GUROBI, cp.CLARABEL, cp.ECOS]
    if solver is not None:
        solver_order = [solver] + solver_order

    solved = False
    for s in solver_order:
        try:
            prob.solve(solver=s, verbose=verbose)
            if prob.status in ("optimal", "optimal_inaccurate") and X.value is not None:
                solved = True
                if verbose:
                    print(f"  Solver: {s}  Status: {prob.status}")
                break
        except Exception:
            continue

    if not solved:
        raise RuntimeError("No available solver could solve the problem.")

    return X.value


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLUSTER RECOVERY (Section 6 of the paper)
# ─────────────────────────────────────────────────────────────────────────────

def recover_clusters(X_opt, tol=1e-6):
    """
    Given the (approximately) optimal solution X_opt, assign points to clusters
    using the tolerance-based scheme from Section 6 of the paper:

        i and j share a cluster iff ||x_i* - x_j*|| <= sqrt(tol)

    Parameters
    ----------
    X_opt : (n, d) solution matrix
    tol   : convergence tolerance used by the solver

    Returns
    -------
    cluster_ids : (n,) integer array  — cluster label for each point
    clusters    : dict  cluster_label -> set of point indices
    """
    n = X_opt.shape[0]
    threshold = np.sqrt(tol)
    cluster_ids = -np.ones(n, dtype=int)
    clusters = {}
    next_id = 0

    remaining = list(range(n))
    while remaining:
        anchor = remaining[0]
        # Find all points within threshold of the anchor
        dists = np.linalg.norm(X_opt[remaining] - X_opt[anchor], axis=1)
        same = [remaining[k] for k, d in enumerate(dists) if d <= threshold]

        for idx in same:
            cluster_ids[idx] = next_id
            remaining.remove(idx)

        clusters[next_id] = set(same)
        next_id += 1

    return cluster_ids, clusters


# ─────────────────────────────────────────────────────────────────────────────
# 4. SCORING METRICS (Section 6 of the paper)
# ─────────────────────────────────────────────────────────────────────────────

def compute_scores(Vm, clusters, n, K):
    """
    Compute s1, s2, s3 as defined in Section 6.

    Parameters
    ----------
    Vm      : dict  m -> set of indices (ground-truth V_m)
    clusters: dict  cluster_label -> set of indices (recovered clusters)
    n       : total number of points
    K       : true number of components

    Returns
    -------
    s1, s2, s3 and the mapping ell: m -> best_cluster_label
    """
    Kprime = len(clusters)

    # Build mapping ell(m) = argmax_{m'} |V_m ∩ R_{m'}|
    ell = {}
    for m in range(K):
        best_label, best_count = -1, -1
        for label, R in clusters.items():
            count = len(Vm[m] & R)
            if count > best_count:
                best_count = count
                best_label = label
        ell[m] = best_label

    # s1: fraction of V_1 ∪ ... ∪ V_K correctly clustered
    union_Vm = set().union(*Vm.values())
    correct_in_Vm = sum(len(Vm[m] & clusters[ell[m]]) for m in range(K))
    s1 = correct_in_Vm / max(len(union_Vm), 1)

    # s2: fraction of all n points correctly clustered
    # We need to know which true component each point belongs to.
    # Use the Vm membership: a point in Vm[m] assigned to cluster ell(m) is correct.
    correct_total = 0
    for m in range(K):
        correct_total += len(Vm[m] & clusters[ell[m]])
    s2 = correct_total / n

    # s3: number of distinct recovered cluster labels used by ell / K
    distinct = len(set(ell.values()))
    s3 = distinct / K

    return s1, s2, s3, ell


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXPERIMENT 1 — Varying Lambda  (mirrors Table 1 in the paper)
# ─────────────────────────────────────────────────────────────────────────────

def experiment1(n=200, verbose=False):
    """
    Experiment 1: vary lambda around lambda* and record s1, s2, s3.

    NOTE: We use n=200 (instead of 1000) to keep runtime tractable with
          open-source solvers.  The paper used n=1000 with a custom ADMM.
    """
    print("=" * 65)
    print("EXPERIMENT 1 — Varying Lambda  (L1 norm, mirrors Table 1)")
    print("=" * 65)

    # ── Parameters ──────────────────────────────────────────────
    d = K = 6
    sigma = 0.0094
    theta = 2.0
    weights = np.ones(K) / K
    # µ_i = e_i  (i-th standard basis vector)
    means = np.eye(K)          # shape (6, 6)

    # lambda* in the L2 paper makes lower-bound ≈ upper-bound on λ.
    # For L1 we use the same formula as a reference point.
    # Lower bound (from eq. 13, adapted):  2*theta*sigma / (F(theta,d)*w_min * n)
    from scipy.stats import chi
    F_theta_d = chi.cdf(theta * np.sqrt(d), df=d)   # CDF of chi with d d.o.f.
    w_min = 1.0 / K
    eps = w_min * F_theta_d / 2
    lambda_lower = 2 * theta * sigma / ((F_theta_d * w_min - eps) * n)
    # Upper bound (from eq. 14):  ||mu_m - mu_m'|| / (2*(n-1))
    # Minimum pairwise distance between means (all ||e_i - e_j|| = sqrt(2))
    min_dist = np.sqrt(2)
    lambda_upper = min_dist / (2 * (n - 1))
    lambda_star = (lambda_lower + lambda_upper) / 2

    print(f"\n  n={n}, d=K={d}, sigma={sigma}, theta={theta}")
    print(f"  F(theta,d) = {F_theta_d:.4f}")
    print(f"  lambda_lower = {lambda_lower:.6f}")
    print(f"  lambda_upper = {lambda_upper:.6f}")
    print(f"  lambda*      = {lambda_star:.6f}\n")

    # Generate data
    A, labels = generate_mixture_of_gaussians(n, d, K, means, sigma, weights)
    Vm = compute_Vm(A, labels, means, np.full(K, sigma), theta, K)
    union_Vm_size = sum(len(v) for v in Vm.values())

    kappas = [0.25, 0.5, 1.0, 2.0, 4.0]
    results = []

    for kappa in kappas:
        lam = kappa * lambda_star
        print(f"  Solving for lambda = {kappa}*lambda* = {lam:.6f} ...", end=" ", flush=True)
        try:
            X_opt = solve_l1_son_clustering(A, lam, verbose=False)
            cluster_ids, clusters = recover_clusters(X_opt)
            s1, s2, s3, ell = compute_scores(Vm, clusters, n, K)
            correct_Vm = int(s1 * union_Vm_size)
            correct_all = int(s2 * n)
            distinct = int(s3 * K)
            print(f"done  (s1={correct_Vm}/{union_Vm_size}, s2={correct_all}/{n}, s3={distinct}/{K})")
            results.append((kappa, lam, correct_Vm, union_Vm_size,
                            correct_all, n, distinct, K))
        except Exception as e:
            print(f"FAILED: {e}")
            results.append((kappa, lam, None, union_Vm_size, None, n, None, K))

    # Print table
    print("\n  ┌────────────┬──────────────────────────┬──────────────────────────┬──────────────────────┐")
    print("  │   lambda   │  s1 (% of Vm recovered)  │  s2 (total % recovered)  │  s3 (distinct clust) │")
    print("  ├────────────┼──────────────────────────┼──────────────────────────┼──────────────────────┤")
    for r in results:
        kappa, lam, c_Vm, tot_Vm, c_all, tot_all, dist, K_ = r
        label = f"{kappa}*λ*"
        s1_str = f"{c_Vm}/{tot_Vm}" if c_Vm is not None else "ERR"
        s2_str = f"{c_all}/{tot_all}" if c_all is not None else "ERR"
        s3_str = f"{dist}/{K_}" if dist is not None else "ERR"
        print(f"  │ {label:^10s} │ {s1_str:^24s} │ {s2_str:^24s} │ {s3_str:^20s} │")
    print("  └────────────┴──────────────────────────┴──────────────────────────┴──────────────────────┘")
    print()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. EXPERIMENT 2 — Varying Sigma  (mirrors Table 2 in the paper)
# ─────────────────────────────────────────────────────────────────────────────

def experiment2(n=200, verbose=False):
    """
    Experiment 2: vary sigma, observe degradation of recovery.
    d=1, K=2, mu1=0, mu2=1, equal weights, theta=1.
    """
    print("=" * 65)
    print("EXPERIMENT 2 — Varying Sigma  (L1 norm, mirrors Table 2)")
    print("=" * 65)

    d, K = 1, 2
    means = np.array([[0.0], [1.0]])
    weights = np.array([0.5, 0.5])
    theta = 1.0
    from scipy.stats import chi
    F_theta_d = chi.cdf(theta * np.sqrt(d), df=d)
    w_min = 0.5
    eps = w_min * F_theta_d / 2

    # sigma* makes lower bound = upper bound on lambda
    # lower: 2*theta*sigma / ((F*w - eps)*n)  =  upper: |mu1-mu2| / (2*(n-1))
    mu_dist = abs(means[1, 0] - means[0, 0])   # = 1.0
    lambda_upper = mu_dist / (2 * (n - 1))
    # sigma* from: 2*theta*sigma* / ((F*w - eps)*n) = lambda_upper
    sigma_star = lambda_upper * (F_theta_d * w_min - eps) * n / (2 * theta)

    print(f"\n  n={n}, d={d}, K={K}, theta={theta}")
    print(f"  sigma* = {sigma_star:.6f},  lambda_max = {lambda_upper:.6f}\n")

    # Fix lambda = lambda_max (does not depend on sigma)
    lam = lambda_upper
    multipliers = [1.0, np.sqrt(2), 2.0, 2 * np.sqrt(2), 4.0]

    results = []
    for mult in multipliers:
        sigma = mult * sigma_star
        A, labels = generate_mixture_of_gaussians(n, d, K, means, sigma, weights)
        Vm = compute_Vm(A, labels, means, np.full(K, sigma), theta, K)
        union_Vm_size = sum(len(v) for v in Vm.values())

        label_str = f"{mult:.3g}σ*"
        print(f"  sigma = {label_str} = {sigma:.6f} ...", end=" ", flush=True)
        try:
            X_opt = solve_l1_son_clustering(A, lam, verbose=False)
            cluster_ids, clusters = recover_clusters(X_opt)
            s1, s2, s3, ell = compute_scores(Vm, clusters, n, K)
            correct_Vm = int(s1 * union_Vm_size)
            correct_all = int(s2 * n)
            distinct = int(s3 * K)
            print(f"done  (s1={correct_Vm}/{union_Vm_size}, s2={correct_all}/{n}, s3={distinct}/{K})")
            results.append((label_str, correct_Vm, union_Vm_size,
                            correct_all, n, distinct, K))
        except Exception as e:
            print(f"FAILED: {e}")
            results.append((label_str, None, union_Vm_size, None, n, None, K))

    print("\n  ┌────────────┬──────────────────────────┬──────────────────────────┬──────────────────────┐")
    print("  │   sigma    │  s1 (% of Vm recovered)  │  s2 (total % recovered)  │  s3 (distinct clust) │")
    print("  ├────────────┼──────────────────────────┼──────────────────────────┼──────────────────────┤")
    for r in results:
        sig_str, c_Vm, tot_Vm, c_all, tot_all, dist, K_ = r
        s1_str = f"{c_Vm}/{tot_Vm}" if c_Vm is not None else "ERR"
        s2_str = f"{c_all}/{tot_all}" if c_all is not None else "ERR"
        s3_str = f"{dist}/{K_}" if dist is not None else "ERR"
        print(f"  │ {sig_str:^10s} │ {s1_str:^24s} │ {s2_str:^24s} │ {s3_str:^20s} │")
    print("  └────────────┴──────────────────────────┴──────────────────────────┴──────────────────────┘")
    print()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. 2-D CLUSTERPATH VISUALIZATION  (d reduced to 2 for illustration)
# ─────────────────────────────────────────────────────────────────────────────

def experiment_clusterpath(n=60, num_lambda=12, verbose=False):
    """
    Trace the L1 clusterpath: solve for a grid of lambda values and
    plot how the centroids x_i* move as lambda increases.
    Uses d=2, K=3 so the result is easy to visualise.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    print("=" * 65)
    print("CLUSTERPATH — L1 fusion, d=2, K=3")
    print("=" * 65)

    d, K = 2, 3
    means = np.array([[0.0, 0.0], [3.0, 0.0], [1.5, 2.6]])
    sigma = 0.4
    weights = np.ones(K) / K

    A, labels = generate_mixture_of_gaussians(n, d, K, means, sigma, weights, seed=7)

    from scipy.stats import chi
    F = chi.cdf(2.0 * np.sqrt(d), df=d)
    w_min = 1 / K
    eps = w_min * F / 2
    lambda_lower = 2 * 2.0 * sigma / ((F * w_min - eps) * n)
    min_dist = np.min([np.linalg.norm(means[i] - means[j])
                       for i, j in itertools.combinations(range(K), 2)])
    lambda_upper = min_dist / (2 * (n - 1))
    lambdas = np.logspace(np.log10(lambda_lower * 0.1),
                          np.log10(lambda_upper * 20),
                          num=num_lambda)

    paths = {i: [] for i in range(n)}
    lambda_vals = []

    for lam in lambdas:
        print(f"  lambda = {lam:.5f} ...", end=" ", flush=True)
        try:
            X_opt = solve_l1_son_clustering(A, lam, verbose=False)
            for i in range(n):
                paths[i].append(X_opt[i].copy())
            lambda_vals.append(lam)
            print("done")
        except Exception as e:
            print(f"skip ({e})")

    # ── Plot ────────────────────────────────────────────────────
    cmap = get_cmap("tab10")
    colors = [cmap(labels[i]) for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0e0e14")
    fig.suptitle("L₁ Sum-of-Norms Clusterpath  (d=2, K=3)",
                 fontsize=15, color="white", fontweight="bold", y=1.01)

    for ax in axes:
        ax.set_facecolor("#0e0e14")
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333")

    # Left: raw data coloured by true component
    ax0 = axes[0]
    for m in range(K):
        mask = labels == m
        ax0.scatter(A[mask, 0], A[mask, 1], c=[cmap(m)],
                    s=30, alpha=0.7, edgecolors="none", label=f"Component {m+1}")
    ax0.scatter(means[:, 0], means[:, 1], c="white", s=120,
                marker="*", zorder=5, label="Means")
    ax0.set_title("Raw Data", color="white", fontsize=12)
    ax0.legend(fontsize=8, labelcolor="white",
               facecolor="#1a1a2e", edgecolor="#333")

    # Right: clusterpath lines
    ax1 = axes[1]
    # Draw raw data faintly
    ax1.scatter(A[:, 0], A[:, 1], c=colors, s=10, alpha=0.2, edgecolors="none")

    # Draw path for each point
    if lambda_vals:
        for i in range(n):
            pts = np.array(paths[i])   # (num_solved, 2)
            if len(pts) > 1:
                ax1.plot(pts[:, 0], pts[:, 1],
                         color=cmap(labels[i]), alpha=0.5, linewidth=0.8)
        # Mark the final (merged) positions
        final_X = np.array([paths[i][-1] for i in range(n)])
        ax1.scatter(final_X[:, 0], final_X[:, 1],
                    c=colors, s=25, zorder=4, edgecolors="white", linewidths=0.3)

    ax1.set_title(f"Clusterpath  (λ from {lambdas[0]:.4f} to {lambdas[-1]:.4f})",
                  color="white", fontsize=12)

    plt.tight_layout()
    out_path = "l1_clusterpath.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  Clusterpath figure saved → {out_path}\n")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 8. COMMENTARY: L1 vs L2 DIFFERENCES
# ─────────────────────────────────────────────────────────────────────────────

COMMENTARY = """
╔══════════════════════════════════════════════════════════════════╗
║            L1  vs  L2  Sum-of-Norms Clustering                  ║
╚══════════════════════════════════════════════════════════════════╝

┌─ Sparsity ─────────────────────────────────────────────────────┐
│ L1 fusion promotes COORDINATE-WISE sparsity: each coordinate    │
│ of x_i* snaps to the corresponding coordinate of a neighbour    │
│ independently.  This makes clusters align with axis-parallel    │
│ hyperplanes, reinforcing the "boxy" geometry.                   │
└────────────────────────────────────────────────────────────────┘

┌─ Geometry ─────────────────────────────────────────────────────┐
│ L2 fusion: x_i* move toward each other along the line joining  │
│ them → clusters are essentially convex hulls.                   │
│ L1 fusion: x_i* move coordinate-by-coordinate → cluster        │
│ boundaries are Manhattan (L1) balls (diamond-shaped in 2-D).   │
│ For spherical Gaussians this means points in the tails that are │
│ far in one coordinate but close in others may merge/split       │
│ differently from L2.                                            │
└────────────────────────────────────────────────────────────────┘

┌─ Lambda thresholds ────────────────────────────────────────────┐
│ The paper's Theorem 3 bounds (eqs. 13-14) use L2 norms.        │
│ For L1, the effective separation and the per-term contribution  │
│ are different.  Empirically you may find:                       │
│   • The window [lambda_lower, lambda_upper] may be wider for L1 │
│     because the L1 distance between cluster centres is >= L2.   │
│   • The critical lambda* may be different — tune accordingly.   │
└────────────────────────────────────────────────────────────────┘

┌─ Computation ──────────────────────────────────────────────────┐
│ L2 version: solved with ADMM at O(n² d) per iteration,          │
│             O(n) iterations → O(n³) total.                      │
│ L1 version: reformulates as an LP (introduce auxiliary vars     │
│             t_{ij} >= ±(x_ik - x_jk)) — interior-point LP      │
│             solvers (MOSEK, GUROBI) handle this efficiently.    │
│             For moderate n, LP is often FASTER than L2 ADMM.   │
└────────────────────────────────────────────────────────────────┘
"""

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(COMMENTARY)

    # Run Experiment 1 (n=200 for speed; paper used n=1000)
    exp1_results = experiment1(n=200, verbose=False)

    # Run Experiment 2
    exp2_results = experiment2(n=200, verbose=False)

    # Run clusterpath (d=2 for visualisation)
    clusterpath_fig = experiment_clusterpath(n=60, num_lambda=10, verbose=False)

    print("\nAll experiments complete.")
    print(f"Clusterpath figure: {clusterpath_fig}")
