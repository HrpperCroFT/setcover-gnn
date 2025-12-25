import torch
import dgl
import random
import numpy as np
from setcover_gnn import (
    generate_set_cover_instance,
    set_cover_to_qubo_qubovert,
    greedy_set_cover,
    verify_set_cover,
    SetCoverGNN,
    create_dgl_graph_from_qubo,
    train_setcover_gnn
)


def main():
    # Configuration
    n = 100
    m = 30
    A = 4.0
    B = 1.5
    seed = 42
    
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate Set Cover instance
    print("Generating Set Cover instance...")
    subsets = generate_set_cover_instance(n, m, coverage_factor=6.0)
    
    # Verify with greedy algorithm
    greedy_solution = greedy_set_cover(n, subsets)
    is_cover, count = verify_set_cover(n, subsets, greedy_solution)
    print(f"Greedy solution is valid: {is_cover}, subsets used: {count}")
    
    # Convert to QUBO
    print("Converting to QUBO...")
    qubo_matrix = set_cover_to_qubo_qubovert(n, subsets, A=A, B=B)
    
    # Create graph
    print("Creating graph...")
    dgl_graph = create_dgl_graph_from_qubo(qubo_matrix)
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    qubo_matrix = qubo_matrix.to(device).to(torch.float32)
    dgl_graph = dgl_graph.to(device)
    
    # Create model
    print("Creating GNN model...")
    model = SetCoverGNN(
        qubo_matrix=qubo_matrix,
        graph=dgl_graph,
        dim_embedding=10,
        hidden_dim=51,
        dropout=0.5,
        learning_rate=1e-3,
        prob_threshold=0.5,
        A=A,
        n_elements=n
    ).to(device)
    
    # Train model
    print("Training GNN...")
    trained_model = train_setcover_gnn(
        model,
        max_epochs=60000,
        patience=100,
        tolerance=1e-4,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )
    
    # Get solution
    solution = trained_model.get_solution()
    solution_np = solution.cpu().numpy()[:m].tolist()
    
    # Verify solution
    is_cover, count = verify_set_cover(n, subsets, solution_np)
    print(f"\nGNN solution is valid: {is_cover}, subsets used: {count}")
    
    return trained_model, solution_np


if __name__ == "__main__":
    model, solution = main()