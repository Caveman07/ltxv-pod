FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Системные зависимости
RUN apt update && apt install -y \
    python3 python3-pip git git-lfs ffmpeg curl && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt clean

# Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Клонируем модели из Hugging Face
RUN git lfs install && \
    git clone https://huggingface.co/Lightricks/LTX-Video-ICLoRA-pose-13b-0.9.7 /app/models/pose && \
    git clone https://huggingface.co/Lightricks/LTX-Video-ICLoRA-canny-13b-0.9.7 /app/models/canny

# Копируем код
COPY app.py /app/app.py
COPY .env.example /app/.env
WORKDIR /app

# Переменные окружения
ENV API_TOKEN=changeme

# Старт сервера
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]