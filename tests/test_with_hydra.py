from git import Repo
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any
import json

import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from setcover_gnn.core import SetCoverSolver, SetCoverProblem
from scripts.msc_parser import load_problem_from_dvc


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
        
        # Создаем MLFlowLogger для всего теста
        self.mlflow_logger = None
        if self.logging_enabled and self.logging_backend == 'mlflow':
            self._setup_mlflow_logger()

    def _setup_mlflow_logger(self):
        """Setup MLflow logger with git commit info."""
        try:
            from lightning.pytorch.loggers import MLFlowLogger
            import sys
            
            # Получаем значения из конфига
            experiment_name = str(self.cfg.logging.mlflow.experiment_name)
            run_name = str(self.cfg.logging.mlflow.run_name)
            tracking_uri = str(self.cfg.logging.mlflow.tracking_uri)
            
            # Преобразуем теги в обычный словарь
            tags = {}
            if self.cfg.logging.mlflow.get('tags'):
                tags = OmegaConf.to_container(self.cfg.logging.mlflow.tags, resolve=True)
                if not isinstance(tags, dict):
                    tags = {}
            
            # Добавляем git commit и версию Python
            git_commit_id = "unknown"
            try:
                repo = Repo(Path(__file__).absolute().parent)
                
                git_commit_id = repo.head.commit.hexsha
            except Exception as e:
                logging.warning(f"Could not get git commit ID: {e}")
            
            tags.update({
                'git_commit': git_commit_id,
                'python_version': sys.version.split()[0],
                'test_run': 'true',
                'framework': 'pytorch-lightning'
            })
            
            # Создаем логгер
            self.mlflow_logger = MLFlowLogger(
                experiment_name=experiment_name,
                run_name=run_name,
                tracking_uri=tracking_uri,
                tags=tags
            )
            
            # Записываем гиперпараметры
            params = self._flatten_dict(OmegaConf.to_container(self.cfg, resolve=True))
            self.mlflow_logger.log_hyperparams(params)
            
            logging.info(f"MLflow logger initialized. Git commit: {git_commit_id}")
            logging.info(f"MLflow tracking URI: {tracking_uri}")
            
        except ImportError:
            logging.warning("MLflow not installed. Logging disabled.")
            self.logging_enabled = False
        except Exception as e:
            logging.warning(f"Failed to setup MLflow logger: {e}")
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
            logging.warning(f"Failed to log metrics to MLflow: {e}")
    
    def _load_problem_from_config(self) -> SetCoverProblem:
        """Загружает проблему в зависимости от конфигурации."""
        if self.cfg.data.source == "generate":
            # Генерация проблемы
            return self.solver.generate_problem(
                n_elements=self.cfg.data.generate.problem.n_elements,
                n_subsets=self.cfg.data.generate.problem.n_subsets,
                coverage_factor=self.cfg.data.generate.problem.get('coverage_factor', 1.5),
                A=self.cfg.model.qubo.A,
                B=self.cfg.model.qubo.B
            )
        elif self.cfg.data.source == "file":
            if not self.cfg.data.file.path:
                raise ValueError("Для source='file' необходимо указать data.file.path")
            
            from setcover_gnn.data.qubo_conversion import set_cover_to_qubo_qubovert
            from setcover_gnn.utils.graph_utils import create_dgl_graph_from_qubo

            n_elements, subsets = load_problem_from_dvc(
                file_path=self.cfg.data.file.path,
                repo_path=self.cfg.dvc.repo_path,
                remote_name=self.cfg.dvc.get('remote_name', 'setcover_gnn_data'),
                use_dvc=self.cfg.data.file.get('use_dvc', True)
            )

            qubo_matrix = set_cover_to_qubo_qubovert(
                n_elements,
                subsets,
                A=self.cfg.model.qubo.A,
                B=self.cfg.model.qubo.B
            )
            
            # Создаем граф
            graph = create_dgl_graph_from_qubo(qubo_matrix)
            
            # Перемещаем на устройство
            qubo_matrix = qubo_matrix.to(self.solver.device).to(self.solver.dtype)
            graph = graph.to(self.solver.device)
            
            return SetCoverProblem(
                n_elements=n_elements,
                subsets=subsets,
                qubo_matrix=qubo_matrix,
                graph=graph
            )
        else:
            raise ValueError(f"Неизвестный источник данных: {self.cfg.data.source}")
    
    def run_test(self) -> Dict[str, Any]:
        """Run a test with the configured parameters."""
        logging.info("Starting test run with Hydra configuration")
        logging.info(f"Configuration:\n{OmegaConf.to_yaml(self.cfg)}")
        
        # Загрузка проблемы
        problem = self._load_problem_from_config()
        
        # Greedy решение (базовая линия)
        greedy_solution, greedy_valid, greedy_count = self.solver.solve_greedy(problem)
        
        # GNN решение
        _, gnn_solution, gnn_metrics = self.solver.solve_gnn(
            problem,
            cfg=self.cfg,
            logger=self.mlflow_logger,  # Передаем логгер в solver
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
                'n_subsets': len(problem.subsets),
                'source': self.cfg.data.source,
                'file': self.cfg.data.get('file', {}).get('path') if self.cfg.data.source == 'file' else None
            },
            'greedy': {
                'selected_count': greedy_count,
                'is_valid': greedy_valid,
                'solution': greedy_solution
            },
            'gnn': {
                'selected_count': gnn_metrics['selected_count'],
                'is_valid': gnn_metrics['is_valid'],
                'solution': gnn_solution
            },
            'comparison': {
                'improvement': greedy_count - gnn_metrics['selected_count'],
                'both_valid': greedy_valid and gnn_metrics['is_valid']
            }
        }
        
        # Сохранение результатов в файл
        output_path = Path(self.cfg.paths.output_dir) / f"{self.cfg.experiment_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Запись артефактов в MLflow
        if self.mlflow_logger is not None:
            try:
                import mlflow
                mlflow.log_artifact(str(output_path))
                logging.info(f"Artifact logged to MLflow: {output_path}")
            except Exception as e:
                logging.warning(f"Failed to log artifact to MLflow: {e}")
        
        logging.info(f"Test completed. Results saved to {output_path}")
        logging.info(f"Greedy: {greedy_count} subsets, valid: {greedy_valid}")
        logging.info(f"GNN: {gnn_metrics['selected_count']} subsets, valid: {gnn_metrics['is_valid']}")
        
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