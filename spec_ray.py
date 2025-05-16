import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
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

# ------------------ Evaluation Metrics ------------------ #
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

# ------------------ Incremental Spectral Clustering via Perturbation ------------------ #
def compute_normalized_laplacian(adj):
    D = np.diag(adj.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))
    L = np.eye(adj.shape[0]) - D_inv_sqrt @ adj @ D_inv_sqrt
    return L

def incremental_spectral_clustering(adj_prev, adj_curr, U_prev, Lambda_prev, k):
    L_prev = compute_normalized_laplacian(adj_prev)
    L_curr = compute_normalized_laplacian(adj_curr)
    delta_L = L_curr - L_prev

    delta_U = np.zeros_like(U_prev)
    for i in range(k):
        for j in range(k):
            if i != j:
                lambda_diff = Lambda_prev[i] - Lambda_prev[j]
                if np.abs(lambda_diff) > 1e-8:
                    delta_U[:, i] += (U_prev[:, j].T @ delta_L @ U_prev[:, i]) / lambda_diff * U_prev[:, j]

    U_curr = U_prev + delta_U
    U_curr = U_curr / (np.linalg.norm(U_curr, axis=1, keepdims=True) + 1e-8)
    labels = KMeans(n_clusters=k, n_init=10).fit_predict(U_curr)
    return labels, U_curr

# ------------------ Main Testing Pipeline ------------------ #
def test_incremental_gsp_vs_kmeans(n=300, k=4, T=30):
    adjacencies, true_communities = generate_dynamic_graph(n, k, T)
    gsp_results = defaultdict(list)
    kmeans_results = defaultdict(list)

    U_prev = None
    Lambda_prev = None
    adj_prev = None

    for t in range(T):
        adj = adjacencies[t]
        true_labels = true_communities[t]

        if t == 0:
            L = compute_normalized_laplacian(adj)
            Lambda_prev, U_prev = eigsh(L, k=k, which='SM')
            U_prev = U_prev / (np.linalg.norm(U_prev, axis=1, keepdims=True) + 1e-8)
            gsp_labels = KMeans(n_clusters=k, n_init=10).fit_predict(U_prev)
        else:
            gsp_labels, U_prev = incremental_spectral_clustering(adj_prev, adj, U_prev, Lambda_prev, k)

        for metric, value in evaluate_clustering(true_labels, gsp_labels, adj).items():
            gsp_results[metric].append(value)

        km_labels = KMeans(n_clusters=k, n_init=10).fit_predict(adj)
        for metric, value in evaluate_clustering(true_labels, km_labels, adj).items():
            kmeans_results[metric].append(value)

        adj_prev = adj

    print("\n=== Final Results ===")
    print(f"{'Metric':<12} {'GSP Mean':<10} {'KMeans Mean':<12} {'Improvement'}")
    for metric in gsp_results:
        gsp_mean = np.mean(gsp_results[metric])
        km_mean = np.mean(kmeans_results[metric])
        impr = (gsp_mean - km_mean) / km_mean * 100 if km_mean != 0 else float('inf')
        print(f"{metric:<12} {gsp_mean:.4f}     {km_mean:.4f}      {impr:+.2f}%")

    return gsp_results, kmeans_results

# ------------------ Run and Plot ------------------ #
gsp_res, km_res = test_incremental_gsp_vs_kmeans(1000,100,50)

plt.figure(figsize=(12, 4))
for i, metric in enumerate(['ARI', 'NMI', 'Modularity'], 1):
    plt.subplot(1, 3, i)
    plt.plot(gsp_res[metric], label='Incremental GSP SC')
    plt.plot(km_res[metric], label='Naive KMeans')
    plt.title(metric)
    plt.xlabel('Snapshot')
    if i == 1:
        plt.ylabel('Score')
    plt.legend()
plt.tight_layout()
plt.show()
