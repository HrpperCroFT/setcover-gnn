import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import dgl
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any
import json

from setcover_gnn.core import SetCoverSolver, SetCoverProblem
from setcover_gnn.models.lightning_module import SetCoverGNN

logger = logging.getLogger(__name__)


class TestRunnerWithHydra:
    """Test runner that uses Hydra configuration."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.solver = SetCoverSolver(
            device=cfg.get('device', 'auto'),
            seed=cfg.get('seed')
        )
        
        # Настройка логирования
        self.logging_enabled = cfg.logging.get('enabled', False)
        self.logging_backend = cfg.logging.get('backend', 'mlflow')
        
        if self.logging_enabled and self.logging_backend == 'mlflow':
            self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow logging."""
        try:
            import mlflow
            mlflow.set_tracking_uri(self.cfg.logging.mlflow.tracking_uri)
            mlflow.set_experiment(self.cfg.logging.mlflow.experiment_name)
            
            # Создаем run
            self.mlflow_run = mlflow.start_run(
                run_name=self.cfg.logging.mlflow.run_name
            )
            
            # Логируем конфигурацию
            mlflow.log_params(self._flatten_dict(OmegaConf.to_container(self.cfg)))
            mlflow.set_tags(self.cfg.logging.mlflow.get('tags', {}))
            
            logger.info(f"MLflow logging enabled at {self.cfg.logging.mlflow.tracking_uri}")
            
        except ImportError:
            logger.warning("MLflow not installed. Logging disabled.")
            self.logging_enabled = False
        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}")
            self.logging_enabled = False
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary for MLflow logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _log_metrics(self, metrics: Dict[str, Any], step: int = None):
        """Log metrics to MLflow."""
        if not self.logging_enabled:
            return
        
        try:
            import mlflow
            
            # Фильтруем только числовые метрики
            numeric_metrics = {
                k: v for k, v in metrics.items() 
                if isinstance(v, (int, float, np.number))
            }
            
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics, step=step)
            
            # Логируем артефакты
            if 'solution' in metrics:
                solution_path = Path(self.cfg.paths.output_dir) / "solution.json"
                solution_path.parent.mkdir(parents=True, exist_ok=True)
                with open(solution_path, 'w') as f:
                    json.dump(metrics.get('solution'), f)
                mlflow.log_artifact(str(solution_path))
                
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")
    
    def run_test(self) -> Dict[str, Any]:
        """Run a test with the configured parameters."""
        logger.info("Starting test run with Hydra configuration")
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(self.cfg)}")
        
        # Генерация проблемы
        problem = self.solver.generate_problem(
            n_elements=self.cfg.data.problem.n_elements,
            n_subsets=self.cfg.data.problem.n_subsets,
            coverage_factor=self.cfg.data.problem.get('coverage_factor', 1.5),
            A=self.cfg.model.qubo.A,
            B=self.cfg.model.qubo.B
        )
        
        # Greedy решение (базовая линия)
        greedy_solution, greedy_metrics = self.solver.solve_greedy(problem)
        
        # GNN решение
        gnn_solution, gnn_metrics = self.solver.solve_gnn(
            problem,
            dim_embedding=self.cfg.model.gnn.dim_embedding,
            hidden_dim=self.cfg.model.gnn.hidden_dim,
            dropout=self.cfg.model.gnn.dropout,
            learning_rate=self.cfg.model.gnn.learning_rate,
            prob_threshold=self.cfg.model.gnn.prob_threshold,
            max_epochs=self.cfg.training.max_epochs,
            patience=self.cfg.training.patience,
            tolerance=self.cfg.training.tolerance
        )
        
        # Сбор результатов
        results = {
            'problem': {
                'n_elements': problem.n_elements,
                'n_subsets': len(problem.subsets)
            },
            'greedy': {
                'selected_count': greedy_metrics['selected_count'],
                'is_valid': greedy_metrics['is_valid']
            },
            'gnn': {
                'selected_count': gnn_metrics['selected_count'],
                'is_valid': gnn_metrics['is_valid'],
                'solution': gnn_solution
            },
            'comparison': {
                'improvement': greedy_metrics['selected_count'] - gnn_metrics['selected_count'],
                'both_valid': greedy_metrics['is_valid'] and gnn_metrics['is_valid']
            }
        }
        
        # Логирование метрик
        metrics_to_log = {
            'greedy_subsets': greedy_metrics['selected_count'],
            'gnn_subsets': gnn_metrics['selected_count'],
            'improvement': results['comparison']['improvement'],
            'gnn_valid': int(gnn_metrics['is_valid']),
            'greedy_valid': int(greedy_metrics['is_valid'])
        }
        
        self._log_metrics(metrics_to_log)
        
        # Сохранение результатов
        output_path = Path(self.cfg.paths.output_dir) / f"{self.cfg.experiment_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if self.logging_enabled:
            import mlflow
            mlflow.log_artifact(str(output_path))
        
        logger.info(f"Test completed. Results saved to {output_path}")
        logger.info(f"Greedy: {greedy_metrics['selected_count']} subsets, valid: {greedy_metrics['is_valid']}")
        logger.info(f"GNN: {gnn_metrics['selected_count']} subsets, valid: {gnn_metrics['is_valid']}")
        
        return results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main test function with Hydra."""
    # Создаем директории
    Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(cfg.paths.log_dir) / "test.log"),
            logging.StreamHandler()
        ]
    )
    
    # Запуск теста
    runner = TestRunnerWithHydra(cfg)
    results = runner.run_test()
    
    return results


if __name__ == "__main__":
    main()