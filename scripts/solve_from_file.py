#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, Any, Optional

import click
import hydra
from omegaconf import DictConfig, OmegaConf

from setcover_gnn.core import SetCoverSolver
from scripts.msc_parser import load_problem_from_dvc
from setcover_gnn.data.qubo_conversion import set_cover_to_qubo_qubovert
from setcover_gnn.utils.graph_utils import create_dgl_graph_from_qubo


def solve_problem_from_file(
    input_file: str,
    repo_path: str,
    remote_name: str,
    use_dvc: bool,
    method: str = "gnn",
    config_path: str = "../configs",
    config_name: str = "config",
    **solver_kwargs
) -> Dict[str, Any]:
    """
    Решает задачу Set Cover из файла.
    
    Args:
        input_file: Путь к файлу .msc в DVC репозитории
        repo_path: Путь к корню DVC репозитория
        remote_name: Имя удаленного хранилища DVC
        use_dvc: Использовать ли DVC для чтения файла
        method: Метод решения ('gnn' или 'greedy')
        config_path: Путь к конфигам Hydra
        config_name: Имя конфига
        **solver_kwargs: Дополнительные параметры для решателя
        
    Returns:
        Словарь с результатами
    """
    # Загружаем конфигурацию
    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name)
    
    # Загружаем проблему из файла через DVC API
    click.echo(f"Загрузка проблемы из {input_file} через DVC...")
    n_elements, subsets = load_problem_from_dvc(
        file_path=input_file,
        repo_path=repo_path,
        remote_name=remote_name,
        use_dvc=use_dvc
    )
    
    # Создаем QUBO матрицу
    click.echo("Создание QUBO матрицы...")
    qubo_matrix = set_cover_to_qubo_qubovert(
        n_elements,
        subsets,
        A=cfg.model.qubo.A,
        B=cfg.model.qubo.B
    )
    
    # Создаем граф
    click.echo("Создание графа...")
    graph = create_dgl_graph_from_qubo(qubo_matrix)
    
    # Создаем проблему
    from setcover_gnn.core import SetCoverProblem
    problem = SetCoverProblem(
        n_elements=n_elements,
        subsets=subsets,
        qubo_matrix=qubo_matrix,
        graph=graph
    )
    
    # Решаем проблему
    click.echo(f"Решение проблемы методом {method}...")
    solver = SetCoverSolver(
        device=cfg.get('device', 'auto'),
        seed=cfg.get('seed')
    )
    
    if method == 'gnn':
        # Используем параметры из конфига для GNN
        _, solution, metrics = solver.solve_gnn(
            problem,
            dim_embedding=cfg.model.gnn.dim_embedding,
            hidden_dim=cfg.model.gnn.hidden_dim,
            dropout=cfg.model.gnn.dropout,
            learning_rate=cfg.model.gnn.learning_rate,
            prob_threshold=cfg.model.gnn.prob_threshold,
            max_epochs=cfg.training.max_epochs,
            patience=cfg.training.patience,
            tolerance=cfg.training.tolerance,
            **solver_kwargs
        )
    else:
        # Жадный алгоритм
        solution, metrics = solver.solve(problem, method=method)
    
    # Формируем результаты
    results = {
        'input_file': input_file,
        'problem': {
            'n_elements': n_elements,
            'n_subsets': len(subsets)
        },
        'method': method,
        'solution': solution,
        'metrics': metrics,
        'config': OmegaConf.to_container(cfg, resolve=True)
    }
    
    return results


@click.command()
@click.option('--input-file', 
              help='Путь к файлу .msc в DVC репозитории. Если не указан, берется из конфига.')
@click.option('--output-file', required=True,
              help='Путь для сохранения результатов (JSON)')
@click.option('--method', default='gnn', type=click.Choice(['gnn', 'greedy']),
              help='Метод решения')
@click.option('--repo-path', 
              help='Путь к корню DVC репозитория. Если не указан, берется из конфига.')
@click.option('--config-path', default='../configs',
              help='Путь к конфигурационным файлам Hydra')
@click.option('--config-name', default='config',
              help='Имя конфигурационного файла')
def main(
    input_file: Optional[str],
    output_file: str,
    method: str,
    repo_path: Optional[str],
    config_path: str,
    config_name: str
):
    """
    Решает задачу Set Cover из файла и сохраняет результат.
    
    Пример использования:
        python scripts/solve_from_file.py --output-file results.json
        python scripts/solve_from_file.py --input-file frb30-15-msc/frb30-15-1.msc --output-file results.json
    """
    # Загружаем конфигурацию
    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name)
    
    # Определяем параметры
    if repo_path is None:
        repo_path = cfg.dvc.get('repo_path', '.')
    
    if input_file is None:
        input_file = cfg.get('data_file', 'frb30-15-msc/frb30-15-1.msc')
    
    use_dvc = cfg.dvc.get('use_dvc', True)
    remote_name = cfg.dvc.get('remote_name', 'setcover_gnn_data')
    
    click.echo(f"Решение задачи из файла: {input_file}")
    click.echo(f"Метод: {method}")
    click.echo(f"Путь к репозиторию: {repo_path}")
    click.echo(f"Удаленное хранилище: {remote_name}")
    click.echo(f"Использовать DVC: {use_dvc}")
    click.echo(f"Конфиг: {config_name}")
    
    # Решаем проблему
    results = solve_problem_from_file(
        input_file=input_file,
        repo_path=repo_path,
        remote_name=remote_name,
        use_dvc=use_dvc,
        method=method,
        config_path=config_path,
        config_name=config_name
    )
    
    # Сохраняем результаты
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    click.echo(f"Результаты сохранены в {output_path}")
    click.echo(f"Количество выбранных подмножеств: {results['metrics']['selected_count']}")
    click.echo(f"Решение валидно: {results['metrics']['is_valid']}")
    
    # Выводим краткую информацию
    click.echo("\n" + "="*50)
    click.echo("КРАТКИЕ РЕЗУЛЬТАТЫ:")
    click.echo(f"Метод: {results['method']}")
    click.echo(f"Элементов: {results['problem']['n_elements']}")
    click.echo(f"Подмножеств: {results['problem']['n_subsets']}")
    click.echo(f"Выбрано подмножеств: {results['metrics']['selected_count']}")
    click.echo(f"Валидность решения: {'ДА' if results['metrics']['is_valid'] else 'НЕТ'}")
    click.echo("="*50)


if __name__ == "__main__":
    main()