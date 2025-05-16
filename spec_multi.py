import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from collections import defaultdict
import matplotlib.pyplot as plt

# ---------------------- Graph Generator ---------------------- #
def generate_dynamic_graph(n, k, T, intra_density=0.7, inter_density=0.05, change_rate=0.1, seed=42):
    np.random.seed(seed)
    adjs, labels = [], []

    base_labels = np.repeat(np.arange(k), n // k)
    if len(base_labels) < n:
        base_labels = np.concatenate([base_labels, np.random.choice(k, n - len(base_labels))])

    for t in range(T):
        if t > 0:
            flip = np.random.rand(n) < change_rate
            new_labels = labels[-1].copy()
            new_labels[flip] = np.random.randint(0, k, size=np.sum(flip))
        else:
            new_labels = base_labels.copy()

        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                prob = intra_density if new_labels[i] == new_labels[j] else inter_density
                if np.random.rand() < prob:
                    adj[i, j] = adj[j, i] = 1

        adjs.append(adj)
        labels.append(new_labels)

    return adjs, labels

# ------------------ Multi-slice Spectral Clustering ------------------ #
def multislice_spectral_clustering(adjacency_list, k, omega=1.0):
    T = len(adjacency_list)
    n = adjacency_list[0].shape[0]
    N = n * T

    L_supra = np.zeros((N, N))
    for t, A in enumerate(adjacency_list):
        D = np.diag(A.sum(axis=1))
        L = D - A
        idx = slice(t*n, (t+1)*n)
        L_supra[idx, idx] = L

    for t in range(T - 1):
        for i in range(n):
            u = t*n + i
            v = (t+1)*n + i
            L_supra[u, v] = L_supra[v, u] = -omega
            L_supra[u, u] += omega
            L_supra[v, v] += omega

    _, eigvecs = eigsh(L_supra, k=k, which='SM')
    norms = np.linalg.norm(eigvecs, axis=1, keepdims=True)
    embeddings = eigvecs / (norms + 1e-8)

    global_labels = KMeans(n_clusters=k, n_init=10).fit_predict(embeddings)
    return [global_labels[t*n:(t+1)*n] for t in range(T)]

# ------------------ Hungarian Label Matching ------------------ #
def match_labels(true_labels, pred_labels):
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(pred_labels)
    contingency = np.zeros((len(unique_true), len(unique_pred)))

    for i, true_label in enumerate(unique_true):
        for j, pred_label in enumerate(unique_pred):
            contingency[i, j] = np.sum((true_labels == true_label) & (pred_labels == pred_label))

    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {unique_pred[j]: unique_true[i] for i, j in zip(row_ind, col_ind)}
    return np.array([mapping[p] for p in pred_labels])

# ---------------------- Evaluation Metrics ---------------------- #
def compute_conductance(adj, labels):
    G = nx.from_numpy_array(adj)
    conductances = []
    for c in np.unique(labels):
        nodes = np.where(labels == c)[0]
        if 0 < len(nodes) < len(labels):
            conductances.append(nx.algorithms.cuts.conductance(G, nodes))
    return np.mean(conductances) if conductances else 0

def evaluate_clustering(true_labels, pred_labels, adj):
    matched_pred = match_labels(true_labels, pred_labels)
    G = nx.from_numpy_array(adj)
    communities = [set(np.where(matched_pred == c)[0]) for c in np.unique(matched_pred)]

    return {
        'ARI': adjusted_rand_score(true_labels, matched_pred),
        'NMI': normalized_mutual_info_score(true_labels, matched_pred),
        'Conductance': compute_conductance(adj, matched_pred),
        'Modularity': nx.algorithms.community.modularity(G, communities)
    }

# ------------------ Main Testing Pipeline ------------------ #
def test_dynamic_clustering(num_nodes=1000, num_communities=50, num_snapshots=30):
    adjacencies, true_communities = generate_dynamic_graph(
        num_nodes, num_communities, num_snapshots,
        intra_density=0.3, inter_density=0.2, change_rate=0.01
    )

    spectral_results = defaultdict(list)
    kmeans_results = defaultdict(list)

    # Run multi-slice spectral clustering once
    multi_spec_labels = multislice_spectral_clustering(adjacencies, num_communities)

    for t in range(num_snapshots):
        adj = adjacencies[t]
        true_labels = true_communities[t]

        # Spectral clustering (multi-slice)
        spec_labels = multi_spec_labels[t]
        for metric, value in evaluate_clustering(true_labels, spec_labels, adj).items():
            spectral_results[metric].append(value)

        # KMeans on adjacency matrix
        km_labels = KMeans(n_clusters=num_communities, n_init=10).fit_predict(adj)
        for metric, value in evaluate_clustering(true_labels, km_labels, adj).items():
            kmeans_results[metric].append(value)

    print("\n=== Final Results ===")
    print(f"{'Metric':<12} {'Spectral':<10} {'KMeans':<10} {'Improvement'}")
    for metric in spectral_results:
        s_mean = np.mean(spectral_results[metric])
        k_mean = np.mean(kmeans_results[metric])
        improvement = (s_mean - k_mean) / k_mean * 100 if k_mean != 0 else float('inf')
        print(f"{metric:<12} {s_mean:.4f}     {k_mean:.4f}    {improvement:+.2f}%")

    return spectral_results, kmeans_results

# ------------------ Run the Pipeline ------------------ #
spectral_res, kmeans_res = test_dynamic_clustering()

# ------------------ Visualization ------------------ #
metrics = ['ARI', 'NMI', 'Modularity']
plt.figure(figsize=(12, 4))
for i, metric in enumerate(metrics, 1):
    plt.subplot(1, 3, i)
    plt.plot(spectral_res[metric], label='Multi-slice Spectral Clustering')
    plt.plot(kmeans_res[metric], label='Naive KMeans')
    plt.title(metric)
    plt.xlabel('Snapshot')
    if i == 1:
        plt.ylabel('Score')
    plt.legend()
plt.tight_layout()
plt.show()
