import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import os
from PIL import Image

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

def save_stacked_data(adjacencies, communities, output_dir="dynamic_graph_data"):
    """Saves stacked 3D adjacencies and 2D communities as single files."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "adjacencies.npy"), adjacencies)
    np.save(os.path.join(output_dir, "communities.npy"), communities)
    print(f"Saved data to {output_dir}/adjacencies.npy and {output_dir}/communities.npy")

def visualize_dynamic_graph(adjacencies, communities, output_file="dynamic_graph.gif"):
    """Generates a GIF from stacked data."""
    cmap = ListedColormap(plt.cm.tab10.colors[:num_communities])
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame):
        ax.clear()
        G = nx.from_numpy_array(adjacencies[frame])
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G, pos, 
            node_color=communities[frame], 
            cmap=cmap, 
            ax=ax,
            node_size=100,
            with_labels=False,
            edge_color="gray"
        )
        ax.set_title(f"Snapshot {frame + 1}/{num_snapshots}")
    
    anim = FuncAnimation(fig, update, frames=num_snapshots, interval=1000)
    anim.save(output_file, writer="pillow", fps=1)
    plt.close()

# Example usage
num_nodes = 1000
num_communities = 50
num_snapshots = 10
intra_density = 0.8
inter_density = 0.1
change_rate = 0.05

# Generate and save
adjacencies, communities = generate_dynamic_graph(
    num_nodes, num_communities, num_snapshots,
    intra_density, inter_density, change_rate
)
save_stacked_data(adjacencies, communities)

# Generate GIF
visualize_dynamic_graph(adjacencies, communities)