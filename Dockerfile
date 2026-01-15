FROM python:3.11-slim

#Evita archivos .pyc y buffers


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

#Dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Codigo
COPY app/ ./app/
COPY artifacts/ ./artifacts/

EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]