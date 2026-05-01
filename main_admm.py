# code developed by : Praveen

import numpy as np
from scipy.stats import chi2
from scipy.spatial.distance import pdist, squareform
import time

# ADMM solver
def admm_convex_clustering(A, lambda_val, weights=None, rho=1.0, tol=1e-6, max_iter=5000):

    n, d = A.shape
    
    X = np.copy(A) 
    Z = np.zeros((n, n, d)) 
    U = np.zeros((n, n, d)) 
    
    if weights is None:
        weights = np.ones((n, n))
    
    denom = 1.0 + rho * n
    
    for iteration in range(max_iter):
        X_prev = np.copy(X)
        
        V = X[:, None, :] - X[None, :, :] + U

        V_norms = np.linalg.norm(V, axis=2)

        threshold = (lambda_val * weights) / rho
        mask = V_norms > threshold
        
        Z.fill(0.0)
        Z[mask] = V[mask] * (1.0 - threshold[mask] / V_norms[mask])[:, np.newaxis]
        Z = 0.5 * (Z - np.transpose(Z, (1, 0, 2)))
        
        sum_terms = np.sum(X[None, :, :] + Z - U, axis=1)
        X = (A + rho * sum_terms) / denom

        diff_X = X[:, None, :] - X[None, :, :]
        U = U + diff_X - Z

        primal_res = np.linalg.norm(diff_X - Z)
        dual_res = np.linalg.norm(X - X_prev)
        
        if primal_res < tol and dual_res < tol:
            print(f"    Converged in {iteration + 1} iterations.")
            break
            
    return X

# cluster retrieval
def extract_clusters(X_opt, tol_threshold=1e-3):
    n = X_opt.shape[0]
    unassigned = set(range(n))
    clusters = []
    
    while unassigned:
        i = unassigned.pop()
        current_cluster = [i]
        to_remove = []
        for j in unassigned:
            if np.linalg.norm(X_opt[i] - X_opt[j]) <= tol_threshold:
                current_cluster.append(j)
                to_remove.append(j)
                
        for j in to_remove:
            unassigned.remove(j)
            
        clusters.append(current_cluster)
        
    return clusters

# evaluation metrics
def calculate_metrics(A, clusters, labels_true, means, sigmas, theta):
    K = len(means)
    n = A.shape[0]
    V = []
    for m in range(K):
        distances = np.linalg.norm(A - means[m], axis=1)
        V_m = set(np.where(distances <= theta * sigmas[m])[0])
        V.append(V_m)
        
    union_V = set.union(*V) if V else set()

    l_map = {}
    for m in range(K):
        best_overlap = -1
        best_cluster_idx = -1
        for idx, R in enumerate(clusters):
            overlap = len(V[m].intersection(set(R)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_cluster_idx = idx
        l_map[m] = best_cluster_idx
    s1_num = sum(len(V[m].intersection(set(clusters[l_map[m]]))) for m in range(K))
    s1 = s1_num / len(union_V) if len(union_V) > 0 else 0
    s1_str = f"{s1_num}/{len(union_V)}"
    s2_num = 0
    for i in range(n):
        true_m = labels_true[i]
        if i in clusters[l_map[true_m]]:
            s2_num += 1
    s2 = s2_num / n
    s2_str = f"{s2_num}/{n}"
    distinct_mapped_clusters = len(set(l_map.values()))
    s3_str = f"{distinct_mapped_clusters}/{K}"

    return s1_str, s2_str, s3_str

# experimental setup
def generate_data(n, K, d, weights, means, sigmas):
    # Generates data from a mixture of Gaussians.
    np.random.seed(42)
    A = np.zeros((n, d))
    labels = np.zeros(n, dtype=int)
    
    for i in range(n):
        m = np.random.choice(K, p=weights)
        labels[i] = m
        A[i] = np.random.normal(loc=means[m], scale=sigmas[m], size=d)
    return A, labels

def run_experiment_1():
    print("\nRunning Experiment 1 (Table 1 Replication)")
    n, d, K = 1000, 6, 6
    weights = np.ones(K) / K
    means = np.eye(K)
    sigma = 0.0094
    sigmas = [sigma] * K
    theta = 2.0
    lambda_star = 7.0e-4
    
    A, labels = generate_data(n, K, d, weights, means, sigmas)
    tol_thresh = np.sqrt(1e-6)
    
    kappas = [0.25, 0.5, 1.0, 2.0, 4.0]
    print(f"{'Lambda (κ * λ*)':<15} | {'s1 (V_m recovered)':<20} | {'s2 (Total recovered)':<20} | {'s3 (Distinct clusters)'}")
    
    for kappa in kappas:
        lmbda = kappa * lambda_star
        X_opt = admm_convex_clustering(A, lmbda, tol=1e-6)
        clusters = extract_clusters(X_opt, tol_thresh)
        s1, s2, s3 = calculate_metrics(A, clusters, labels, means, sigmas, theta)
        
        label = f"{kappa} * λ*"
        print(f"{label:<15} | {s1:<20} | {s2:<20} | {s3}")

def run_experiment_2():
    print("\nRunning Experiment 2 (Table 2 Replication)")
    n, d, K = 1000, 1, 2
    weights = [0.5, 0.5]
    means = np.array([[0.0], [1.0]])
    theta = 1.0

    lambda_max = 1.0 / (2 * (n - 1))
    F_val = chi2.cdf(theta**2, df=d) 
    epsilon = 0.0
    
    sigma_star = (lambda_max * (F_val * weights[0] - epsilon) * n) / (2 * theta)
    tol_thresh = np.sqrt(1e-6)
    
    multipliers = [1.0, np.sqrt(2), 2.0, 2.0 * np.sqrt(2), 4.0]
    labels_sigma = ["σ*", "2^(1/2) * σ*", "2 * σ*", "2^(3/2) * σ*", "4 * σ*"]
    
    print(f"{'Sigma multiplier':<18} | {'s1 (V_m recovered)':<20} | {'s2 (Total recovered)':<20} | {'s3 (Distinct clusters)'}")
    
    for mult, label in zip(multipliers, labels_sigma):
        current_sigma = mult * sigma_star
        sigmas = [current_sigma, current_sigma]
        A, labels = generate_data(n, K, d, weights, means, sigmas)
        
        X_opt = admm_convex_clustering(A, lambda_max, tol=1e-6)
        clusters = extract_clusters(X_opt, tol_thresh)
        s1, s2, s3 = calculate_metrics(A, clusters, labels, means, sigmas, theta)
        
        print(f"{label:<18} | {s1:<20} | {s2:<20} | {s3}")

def run_experiment_3():
    print("\nRunning Experiment 3 (Table 3 Replication)")
    n, d, K = 1000, 6, 6
    weights_prob = np.ones(K) / K
    means = np.eye(K)
    sigma = 0.0094
    sigmas = [sigma] * K
    theta = 2.0
    lambda_star = 7.0e-4 
    
    A, labels = generate_data(n, K, d, weights_prob, means, sigmas)
    tol_thresh = np.sqrt(1e-6)
    
    phi_values = [500, 1000, 1500, 2000]
    
    print(f"{'Phi':<10} | {'s1 (V_m recovered)':<20} | {'s2 (Total recovered)':<20} | {'s3 (Distinct clusters)'}")
    
    # Precompute pairwise distances squared for weights
    dist_sq = squareform(pdist(A, metric='sqeuclidean'))
    
    for phi in phi_values:
        W = np.exp(-phi * dist_sq)
        
        X_opt = admm_convex_clustering(A, lambda_star, weights=W, tol=1e-6)
        clusters = extract_clusters(X_opt, tol_thresh)
        s1, s2, s3 = calculate_metrics(A, clusters, labels, means, sigmas, theta)
        
        print(f"{phi:<10} | {s1:<20} | {s2:<20} | {s3}")

if __name__ == "__main__":
    start_time = time.time()
    
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    
    print(f"\nExecution finished in {(time.time() - start_time):.2f} seconds.")