#!/usr/bin/env python3
"""Command-line script for training Set Cover GNN."""

import argparse
import torch
from setcover_gnn import SetCoverSolver


def parse_args():
    parser = argparse.ArgumentParser(description='Train GNN for Set Cover problem')
    parser.add_argument('--n-elements', type=int, default=100, 
                       help='Number of elements in universal set')
    parser.add_argument('--n-subsets', type=int, default=None,
                       help='Number of subsets (default: random)')
    parser.add_argument('--coverage-factor', type=float, default=1.5,
                       help='Coverage factor for generation')
    parser.add_argument('--dim-embedding', type=int, default=10,
                       help='Dimension of node embeddings')
    parser.add_argument('--hidden-dim', type=int, default=51,
                       help='Hidden dimension of GNN')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--max-epochs', type=int, default=60000,
                       help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=100,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Set Cover GNN Training")
    print(f"=====================")
    print(f"Elements: {args.n_elements}")
    print(f"Subsets: {args.n_subsets or 'random'}")
    print(f"Coverage factor: {args.coverage_factor}")
    print(f"Device: {args.device or 'auto'}")
    
    # Initialize solver
    solver = SetCoverSolver(device=args.device, seed=args.seed)
    
    # Generate problem
    problem = solver.generate_problem(
        n_elements=args.n_elements,
        n_subsets=args.n_subsets,
        coverage_factor=args.coverage_factor
    )
    
    # Solve with greedy baseline
    greedy_solution, greedy_metrics = solver.solve(problem, method='greedy')
    print(f"\nGreedy solution: {greedy_metrics['selected_count']} subsets")
    
    # Solve with GNN
    print("\nTraining GNN...")
    gnn_solution, gnn_metrics = solver.solve(
        problem,
        method='gnn',
        dim_embedding=args.dim_embedding,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience
    )
    
    print(f"\nResults:")
    print(f"  Greedy: {greedy_metrics['selected_count']} subsets, valid: {greedy_metrics['is_valid']}")
    print(f"  GNN:    {gnn_metrics['selected_count']} subsets, valid: {gnn_metrics['is_valid']}")
    
    if gnn_metrics['is_valid'] and greedy_metrics['is_valid']:
        improvement = greedy_metrics['selected_count'] - gnn_metrics['selected_count']
        print(f"  Improvement: {improvement} subsets")
    
    # Save if requested
    if args.output:
        import json
        results = {
            'problem': {
                'n_elements': problem.n_elements,
                'n_subsets': len(problem.subsets)
            },
            'greedy': greedy_metrics,
            'gnn': gnn_metrics
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()