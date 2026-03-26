FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 80

# Явно указываем Python-путь, чтобы он видел папку serve
ENV PYTHONPATH=/app
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "80"]
