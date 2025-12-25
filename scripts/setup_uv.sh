#!/bin/bash
set -e

echo "Установка uv..."
pip install uv

echo "Создание виртуального окружения и установка зависимостей..."
uv venv
source .venv/bin/activate

echo "Установка PyTorch и DGL..."
uv pip install \
    torch==2.3.0 \
    torchvision==0.18.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

uv pip install \
    dgl==2.4.0+cu121 \
    --extra-index-url https://data.dgl.ai/wheels/cu121/repo.html

echo "Установка проекта и зависимостей..."
uv pip install -e ".[dev,test]"

echo "Готово! Виртуальное окружение активировано."