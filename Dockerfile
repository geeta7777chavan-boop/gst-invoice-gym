# ======================
# OpenEnv-InvoiceGym Dockerfile
# ======================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY server/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY . .

# Expose ports
# 8000 → FastAPI server (required by OpenEnv)
# 7860 → Gradio demo (for visualization)
EXPOSE 8000

# Health check (optional but good practice)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health')" || exit 1

# Default command: Run the FastAPI server (main requirement for OpenEnv)
# You can run Gradio separately with: python demo/app.py
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
