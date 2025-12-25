FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update
RUN apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    git \
    wget \
    curl \
    python3.11-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем uv
RUN pip3 install --no-cache-dir uv==0.4.24

WORKDIR /workspace

# Копируем файлы проекта для кэширования зависимостей
COPY pyproject.toml README.md ./

# 1. СОЗДАЁМ виртуальное окружение
RUN uv venv .venv

# Активируем виртуальное окружение для последующих команд
ENV PATH="/workspace/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/workspace/.venv"

# 2. Устанавливаем PyTorch с нужными индексами
RUN pip install \
    torch==2.3.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip uninstall torchdata
RUN pip install torchdata==0.8.0

RUN pip install \
    PyYAML \
    pydantic \
    setuptools

# 3. Устанавливаем DGL с официального репозитория
RUN pip install \
    dgl==2.2.1+cu121 \
    --find-links https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

# 4. Устанавливаем основной проект и зависимости групп dev, test
RUN uv pip install --no-build-isolation -e ".[dev,test]"

# Копируем остальной код проекта
COPY . .

# Проверяем установку (уже в виртуальном окружении)
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
RUN python3 -c "import dgl; print(f'DGL: {dgl.__version__}')"
RUN python3 -c "import lightning as pl; print(f'Lightning: {pl.__version__}')"
RUN python3 -c "import click; print(f'Click: {click.__version__}')"

EXPOSE 8080

# Команда по умолчанию с активированным окружением
CMD ["/bin/bash", "-c", "source /workspace/.venv/bin/activate && /bin/bash"]