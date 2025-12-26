#!/usr/bin/env python3
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import gdown
import dvc.api
from dvc.repo import Repo
import logging

logger = logging.getLogger(__name__)


def download_from_gdrive(remote_url: str, output_dir: Path) -> None:
    """
    Скачивает папку с данными из Google Drive.
    
    Args:
        remote_url: URL Google Drive папки
        output_dir: Локальная папка для сохранения
    """
    logger.info(f"Загрузка данных из {remote_url} в {output_dir}...")
    
    # Создаем папку, если не существует
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Извлекаем ID папки из URL
    folder_id = None
    if "/folders/" in remote_url:
        # URL вида: https://drive.google.com/drive/folders/1A2B3C4D...
        parts = remote_url.split("/folders/")
        if len(parts) > 1:
            folder_id = parts[1].split("?")[0].split("/")[0]
    elif "id=" in remote_url:
        # URL с параметром id
        folder_id = remote_url.split("id=")[1].split("&")[0]
    
    if folder_id:
        logger.info(f"ID папки Google Drive: {folder_id}")
        
        # Скачиваем всю папку
        gdown.download_folder(
            id=folder_id,
            output=str(output_dir),
            quiet=False,
            use_cookies=True
        )
        logger.info(f"Данные успешно загружены в {output_dir}")
        
        # Проверяем, что файлы были скачаны
        files = list(output_dir.rglob("*"))
        logger.info(f"Скачано файлов: {len(files)}")
        
    else:
        raise ValueError(f"Не удалось извлечь ID папки из URL: {remote_url}")


def init_dvc_repo(repo_path: Path, remote_name: str = "setcover_gnn_data") -> Repo:
    """
    Инициализирует DVC репозиторий и настраивает удаленное хранилище.
    
    Args:
        repo_path: Путь к корню репозитория
        remote_name: Имя удаленного хранилища
        
    Returns:
        Инициализированный DVC репозиторий
    """
    logger.info(f"Инициализация DVC репозитория в {repo_path}...")
    
    # Проверяем, инициализирован ли уже DVC
    dvc_dir = repo_path / ".dvc"
    
    if dvc_dir.exists():
        logger.info("DVC уже инициализирован в этом репозитории")
        repo = Repo(repo_path)
    else:
        # Инициализируем DVC без Git
        repo = Repo.init(repo_path, no_scm=True, force=True)
        logger.info("DVC успешно инициализирован")
    
    # Настраиваем удаленное хранилище
    try:
        # Добавляем локальное хранилище (в папке data)
        with repo.config.edit() as conf:
            data_dir = repo_path / "data"
            conf["remote"][remote_name] = {"url": str(data_dir)} 
            
            # Устанавливаем как хранилище по умолчанию
            conf["core"]["remote"] = remote_name
            
            logger.info(f"Удаленное хранилище настроено: {remote_name} -> {data_dir}")
            
            # Сохраняем конфигурацию
            repo.scm.add([".dvc/config"])
        
    except Exception as e:
        logger.warning(f"Не удалось настроить удаленное хранилище: {e}")
    
    return repo


def setup_dvc_storage(repo: Repo, data_dir: Path, format: str) -> None:
    """
    Настраивает хранилище DVC и добавляет файлы.
    
    Args:
        repo: DVC репозиторий
        data_dir: Папка с данными
    """
    logger.info(f"Настройка DVC хранилища для {data_dir}...")
    
    # Добавляем все файлы в DVC
    logger.info("Добавление файлов в DVC...")
    
    # Рекурсивно добавляем все файлы
    added_files = []
    for file_path in data_dir.rglob("*"):
        if file_path.is_file() and file_path.name.endswith(format):
            try:
                # Получаем относительный путь от корня репозитория
                rel_path = file_path.relative_to(repo.root_dir)
                
                # Добавляем файл в DVC
                repo.add(str(rel_path))
                added_files.append(str(rel_path))
                logger.debug(f"Добавлен файл: {rel_path}")
                
            except ValueError:
                # Файл не находится внутри репозитория
                logger.warning(f"Файл вне репозитория: {file_path}")
            except Exception as e:
                logger.error(f"Ошибка при добавлении файла {file_path}: {e}")
    
    logger.info(f"Добавлено файлов в DVC: {len(added_files)}")
    
    if added_files:
        # Фиксируем изменения
        repo.scm.add(["*.dvc", ".dvcignore", ".dvc/config"])
        logger.info("Изменения зафиксированы в DVC")
        
        # Отправляем в удаленное хранилище
        try:
            repo.push()
            logger.info("Данные отправлены в удаленное хранилище")
        except Exception as e:
            logger.warning(f"Не удалось отправить данные в удаленное хранилище: {e}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Основная функция загрузки данных.
    
    Args:
        cfg: Конфигурация Hydra
    """
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Конфигурация загрузки данных:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Получаем путь к корню репозитория
    repo_path = Path(cfg.dvc.repo_path).resolve()
    if not repo_path.exists():
        logger.error(f"Путь к репозиторию не существует: {repo_path}")
        sys.exit(1)
    
    logger.info(f"Корень репозитория: {repo_path}")
    
    # Определяем папку для данных
    data_dir = repo_path / "data"
    
    # Проверяем, нужно ли загружать данные
    if cfg.dvc.get("remote_url"):
        remote_url = cfg.dvc.remote_url
        logger.info(f"URL для загрузки: {remote_url}")
        
        # Скачиваем данные
        try:
            download_from_gdrive(remote_url, data_dir)
        except Exception as e:
            logger.error(f"Ошибка при скачивании данных: {e}")
            sys.exit(1)
    else:
        logger.info("remote_url не указан, пропускаем загрузку данных")
        if not data_dir.exists():
            logger.error(f"Папка данных не существует: {data_dir}")
            sys.exit(1)

    try:
        repo = init_dvc_repo(repo_path, cfg.dvc.get("remote_name", "setcover_gnn_data"))

        if data_dir.exists():
            setup_dvc_storage(repo, data_dir, cfg.data.file.get("format"))

        logger.info("Информация о DVC репозитории:")
        logger.info(f"Корневая папка: {repo.root_dir}")
        logger.info(f"Удаленное хранилище: {repo.config.get('core', {}).get('remote')}")

        test_file = cfg.data.file.get("path")
        try:
            content = dvc.api.read(
                path=test_file,
                repo=str(repo_path),
                remote=cfg.dvc.get("remote_name", "setcover_gnn_data")
            )
            logger.info(f"Файл доступен через DVC API: {test_file}")
        except Exception as e:
            logger.warning(f"Файл не доступен через DVC API: {test_file} - {e}")
        
    except Exception as e:
        logger.error(f"Ошибка при инициализации DVC: {e}")
        sys.exit(1)
    
    logger.info("Загрузка данных и инициализация DVC завершены успешно!")
    logger.info(f"Данные доступны в: {data_dir}")
    logger.info("Для работы с DVC используйте команды:")
    logger.info(f"  cd {repo_path}")
    logger.info("  dvc status  # Проверить статус")
    logger.info("  dvc push    # Отправить изменения в удаленное хранилище")
    logger.info("  dvc pull    # Получить изменения из удаленного хранилища")


if __name__ == "__main__":
    main()