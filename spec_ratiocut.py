import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

# Function to generate dynamic graph
def generate_dynamic_graph(
    num_nodes, 
    num_communities, 
    num_snapshots, 
    intra_density=0.8, 
    inter_density=0.1, 
    change_rate=0.05,
    seed=42
):
    """Generates dynamic graph with stacked outputs."""
    np.random.seed(seed)
    
    # Initialize communities with non-uniform probabilities
    community_probs = np.random.dirichlet(np.ones(num_communities))
    communities = np.random.choice(num_communities, size=num_nodes, p=community_probs)
    
    # Preallocate arrays
    adjacencies = np.zeros((num_snapshots, num_nodes, num_nodes), dtype=np.int8)
    community_assignments = np.zeros((num_snapshots, num_nodes), dtype=np.int8)
    
    for t in range(num_snapshots):
        # Create adjacency matrix with community structure
        adj = np.zeros((num_nodes, num_nodes), dtype=np.int8)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if communities[i] == communities[j]:
                    if np.random.rand() < intra_density:
                        adj[i, j] = adj[j, i] = 1
                else:
                    if np.random.rand() < inter_density:
                        adj[i, j] = adj[j, i] = 1
        
        # Store snapshot data
        adjacencies[t] = adj
        community_assignments[t] = communities.copy()
        
        # Evolve communities
        for i in range(num_nodes):
            if np.random.rand() < change_rate:
                new_community = np.random.choice(num_communities)
                # Ensure high connectivity with new community
                for j in range(num_nodes):
                    if communities[j] == new_community and np.random.rand() < intra_density:
                        adj[i, j] = adj[j, i] = 1
                communities[i] = new_community
    
    return adjacencies, community_assignments

# Hungarian matching for label alignment
def hungarian_match(true_labels, pred_labels):
    D = max(pred_labels.max(), true_labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(pred_labels.size):
        w[pred_labels[i], true_labels[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    new_pred_labels = np.zeros_like(pred_labels)
    for i, j in ind:
        new_pred_labels[pred_labels == i] = j
    return new_pred_labels

# Conductance calculation
def conductance(G, labels):
    cut = 0
    volume = 0
    for cluster in np.unique(labels):
        nodes = np.where(labels == cluster)[0]
        volume += sum(dict(G.degree(nodes)).values())
        cut += nx.cut_size(G, nodes)
    return cut / volume if volume > 0 else 0

# Modularity calculation
def modularity(G, labels):
    communities = [list(np.where(labels == c)[0]) for c in np.unique(labels)]
    return nx.algorithms.community.quality.modularity(G, communities)

# Dynamic Ratio Cut with Temporal Smoothness
def dynamic_ratio_cut_temporal(adj_matrices, num_communities, lambd=1.0, max_iter=20):
    """
    Implements the Dynamic Ratio Cut with Temporal Smoothness optimization:
    min_{Y_t} sum_t Tr(Y_t^T L_t Y_t) + lambda sum_t ||Y_t - Y_{t-1}||_F^2
    """
    num_snapshots, num_nodes, _ = adj_matrices.shape
    
    # Initialize embeddings with spectral embeddings
    Y = np.zeros((num_snapshots, num_nodes, num_communities))
    for t in range(num_snapshots):
        adj = adj_matrices[t].astype(float)
        L = np.diag(adj.sum(axis=1)) - adj  # Laplacian matrix
        eigvals, eigvecs = eigsh(L, k=num_communities, which='SM')
        Y[t] = eigvecs
    
    # Iterative optimization
    for iteration in range(max_iter):
        Y_prev = Y.copy()
        
        # Update each Y_t
        for t in range(num_snapshots):
            # Compute current Laplacian
            adj = adj_matrices[t].astype(float)
            L = np.diag(adj.sum(axis=1)) - adj
            
            # Build the system matrices
            A = L.copy()
            B = np.zeros((num_nodes, num_communities))
            
            # Add temporal smoothness terms
            if t > 0:
                A += lambd * np.eye(num_nodes)
                B += lambd * Y[t-1]
            if t < num_snapshots - 1:
                A += lambd * np.eye(num_nodes)
                B += lambd * Y_prev[t+1]
            
            # Solve the linear system A*Y_t = B
            Y[t], _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        
        # Check convergence
        diff = np.linalg.norm(Y - Y_prev, 'fro')
        if diff < 1e-4:
            print(f"Converged after {iteration+1} iterations")
            break
    
    # Normalize embeddings for clustering
    for t in range(num_snapshots):
        norms = np.linalg.norm(Y[t], axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        Y[t] = Y[t] / norms
    
    # Cluster the embeddings
    clusterings = []
    for t in range(num_snapshots):
        kmeans = KMeans(n_clusters=num_communities, random_state=42, n_init=10)
        clusterings.append(kmeans.fit_predict(Y[t]))
    
    return np.array(clusterings)

# Naive KMeans clustering
def naive_kmeans(adj_matrices, num_communities):
    clusterings = []
    for adj in adj_matrices:
        kmeans = KMeans(n_clusters=num_communities, random_state=42, n_init=10)
        clusterings.append(kmeans.fit_predict(adj))
    return np.array(clusterings)

# Evaluation function
def evaluate_clustering(true_labels, pred_labels, adj_matrices):
    ari_scores = []
    nmi_scores = []
    conductance_scores = []
    modularity_scores = []
    num_snapshots = true_labels.shape[0]
    
    for t in range(num_snapshots):
        true = true_labels[t]
        pred = hungarian_match(true, pred_labels[t])
        ari_scores.append(adjusted_rand_score(true, pred))
        nmi_scores.append(normalized_mutual_info_score(true, pred))
        G = nx.from_numpy_array(adj_matrices[t])
        conductance_scores.append(conductance(G, pred))
        modularity_scores.append(modularity(G, pred))
    
    return np.mean(ari_scores), np.mean(nmi_scores), np.mean(conductance_scores), np.mean(modularity_scores)

# Main execution
def main():
    # Generate dynamic graph
    num_nodes = 100
    num_communities = 4
    num_snapshots = 10
    adj_matrices, true_communities = generate_dynamic_graph(
        num_nodes, 
        num_communities, 
        num_snapshots,
        change_rate=0.2  # Higher change rate to test temporal smoothness
    )

    # Perform clustering with dynamic ratio cut with temporal smoothness
    lambd = 10.0  # Temporal smoothness parameter
    pred_dynamic_ratio_cut = dynamic_ratio_cut_temporal(adj_matrices, num_communities, lambd=lambd)

    # Perform clustering with naive kmeans
    pred_naive_kmeans = naive_kmeans(adj_matrices, num_communities)

    # Evaluate
    ari_drc, nmi_drc, cond_drc, mod_drc = evaluate_clustering(
        true_communities, pred_dynamic_ratio_cut, adj_matrices
    )
    ari_km, nmi_km, cond_km, mod_km = evaluate_clustering(
        true_communities, pred_naive_kmeans, adj_matrices
    )

    # Print results
    print(f"Dynamic Ratio Cut with Temporal Smoothness - ARI: {ari_drc:.3f}, NMI: {nmi_drc:.3f}, Conductance: {cond_drc:.3f}, Modularity: {mod_drc:.3f}")
    print(f"Naive KMeans - ARI: {ari_km:.3f}, NMI: {nmi_km:.3f}, Conductance: {cond_km:.3f}, Modularity: {mod_km:.3f}")

    # Plot results
    metrics = ['ARI', 'NMI', 'Conductance', 'Modularity']
    drc_scores = [ari_drc, nmi_drc, cond_drc, mod_drc]
    km_scores = [ari_km, nmi_km, cond_km, mod_km]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 7))
    rects1 = ax.bar(x - width/2, drc_scores, width, label='Dynamic Ratio Cut', color='deepskyblue')
    rects2 = ax.bar(x + width/2, km_scores, width, label='Naive KMeans', color='peachpuff')

    ax.set_ylabel('Scores')
    ax.set_xlabel('Metrics')
    ax.set_title('Comparison of Clustering Methods on Dynamic Graph with Temporal Smoothness')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
