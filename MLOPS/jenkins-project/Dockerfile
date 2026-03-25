FROM python:3.10-slim
WORKDIR /app

# копируем манифест и ставим зависимости
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

# копируем код
COPY . /app

EXPOSE 80

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "80"]
