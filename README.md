# LTXV FastAPI Pod (RunPod-ready)

## 🚀 Возможности:
- Генерация видео с LTXV (или mock)
- Cloudflare R2 загрузка
- Webhook при завершении
- API `/generate`, `/status`, `/shutdown`

## 🔧 Запуск локально (mock):
```bash
cp .env.example .env
export MOCK_MODE=true
uvicorn app:app --reload
```

## 🐳 Сборка Docker:
```bash
docker build -t ltxv-pod .
```

## ✅ Переменные окружения (.env)
- `API_TOKEN`: авторизация
- `MOCK_MODE`: true / false
- `MODEL_NAME`: pose / canny
- `R2_*`: Cloudflare R2
- `WEBHOOK_URL`: опциональный webhook
