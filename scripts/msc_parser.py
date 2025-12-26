from pathlib import Path
from typing import List, Set, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def parse_msc_content(content: str) -> Tuple[int, List[Set[int]]]:
    """
    Парсит содержимое файла формата .msc.
    
    Формат файла:
    p set <n_elements> <n_subsets>
    s <element1> <element2> ... <elementK>
    
    Args:
        content: Содержимое файла .msc
        
    Returns:
        Кортеж (n_elements, subsets)
        
    Raises:
        ValueError: Если формат файла неверный
    """
    subsets = []
    n_elements = None
    n_subsets = None
    
    lines = content.strip().split('\n')
    
    # Обрабатываем строки
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('c'):
            continue  # Пропускаем пустые строки и комментарии
        
        parts = line.split()
        
        if parts[0] == 'p' and parts[1] == 'set':
            # Строка с параметрами
            if len(parts) != 4:
                raise ValueError(
                    f"Строка {line_num}: Неверный формат строки параметров: {line}"
                )
            try:
                n_elements = int(parts[2])
                n_subsets = int(parts[3])
                logger.debug(f"Параметры: {n_elements} элементов, {n_subsets} подмножеств")
            except ValueError:
                raise ValueError(f"Строка {line_num}: Неверные числовые значения: {line}")
        
        elif parts[0] == 's':
            # Строка с подмножеством
            try:
                subset_elements = {int(x) for x in parts[1:]}
                subsets.append(subset_elements)
                logger.debug(f"Подмножество {len(subsets)}: {len(subset_elements)} элементов")
            except ValueError:
                raise ValueError(f"Строка {line_num}: Неверный формат подмножества: {line}")
        
        else:
            raise ValueError(f"Строка {line_num}: Неизвестный тип строки: {line}")
    
    # Проверяем корректность
    if n_elements is None or n_subsets is None:
        raise ValueError("Файл не содержит строку с параметрами (p set ...)")
    
    if len(subsets) != n_subsets:
        raise ValueError(
            f"Количество подмножеств не совпадает: ожидалось {n_subsets}, "
            f"получено {len(subsets)}"
        )
    
    # Проверяем, что элементы в правильном диапазоне
    all_elements = set()
    for i, subset in enumerate(subsets, 1):
        for elem in subset:
            if not 1 <= elem <= n_elements:
                raise ValueError(
                    f"Подмножество {i}: элемент {elem} вне диапазона 1..{n_elements}"
                )
        all_elements.update(subset)
    
    # Проверяем покрытие всех элементов
    if all_elements != set(range(1, n_elements + 1)):
        missing = set(range(1, n_elements + 1)) - all_elements
        raise ValueError(f"Не все элементы покрыты: отсутствуют {missing}")
    
    logger.info(f"Успешно распарсено: {n_elements} элементов, {n_subsets} подмножеств")
    return n_elements, subsets


def load_problem_from_dvc(
    file_path: str,
    repo_path: str = ".",
    remote_name: str = "setcover_gnn_data",
    use_dvc: bool = True
) -> Tuple[int, List[Set[int]]]:
    """
    Загружает проблему из файла через DVC API.
    
    Args:
        file_path: Относительный путь к файлу в DVC репозитории
        repo_path: Путь к корню DVC репозитория
        remote_name: Имя удаленного хранилища DVC
        use_dvc: Использовать ли DVC для чтения файла
        
    Returns:
        Кортеж (n_elements, subsets)
    """
    if use_dvc:
        # Используем DVC API для чтения файла
        try:
            import dvc.api
            
            logger.info(f"Чтение файла через DVC API: {file_path}")
            
            # Читаем содержимое файла через DVC API
            content = dvc.api.read(
                path=file_path,
                repo=repo_path,
                remote=remote_name,
                mode='r',
                encoding='utf-8'
            )
            
            # Парсим содержимое
            return parse_msc_content(content)
            
        except ImportError:
            logger.error("DVC API не доступен. Установите dvc: pip install dvc")
            raise
        except Exception as e:
            logger.error(f"Ошибка при чтении файла через DVC API: {e}")
            raise
    
    else:
        # Режим без DVC (только для тестирования)
        logger.warning("Используется режим без DVC (только для тестирования)")
        
        # Строим полный путь
        full_path = Path(repo_path) / file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Файл не найден: {full_path}")
        
        # Читаем файл обычным способом
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return parse_msc_content(content)