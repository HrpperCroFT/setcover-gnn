import torch
import numpy as np
from setcover_gnn import SetCoverSolver


def main():
    """Advanced example with multiple runs and comparison."""
    
    # Initialize solver with seed for reproducibility
    solver = SetCoverSolver(device='cuda', seed=42)
    
    # Generate problem
    print("Generating Set Cover problem...")
    problem = solver.generate_problem(
        n_elements=100,
        n_subsets=30,
        coverage_factor=2.0,
        A=4.0,
        B=1.5
    )
    
    print(f"Problem: {problem.n_elements} elements, {len(problem.subsets)} subsets")
    
    # Solve with greedy (baseline)
    print("\n--- Greedy Solution ---")
    greedy_solution, greedy_metrics = solver.solve(problem, method='greedy')
    print(f"Valid: {greedy_metrics['is_valid']}")
    print(f"Subsets used: {greedy_metrics['selected_count']}")
    
    # Solve with GNN
    print("\n--- GNN Solution ---")
    gnn_solution, gnn_metrics = solver.solve(
        problem,
        method='gnn',
        dim_embedding=10,
        hidden_dim=51,
        learning_rate=1e-3,
        max_epochs=20000,
        patience=50,
        tolerance=1e-4
    )
    
    print(f"Valid: {gnn_metrics['is_valid']}")
    print(f"Subsets used: {gnn_metrics['selected_count']}")
    
    # Compare solutions
    print("\n--- Comparison ---")
    print(f"Greedy uses {greedy_metrics['selected_count']} subsets")
    print(f"GNN uses {gnn_metrics['selected_count']} subsets")
    
    if gnn_metrics['is_valid']:
        improvement = greedy_metrics['selected_count'] - gnn_metrics['selected_count']
        print(f"GNN improvement: {improvement} subsets fewer")
    
    # Save results
    results = {
        'greedy_solution': greedy_solution,
        'greedy_metrics': greedy_metrics,
        'gnn_solution': gnn_solution,
        'gnn_metrics': gnn_metrics,
        'problem_size': (problem.n_elements, len(problem.subsets))
    }
    
    return results


if __name__ == "__main__":
    results = main()