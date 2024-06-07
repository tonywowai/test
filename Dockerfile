FROM python:3.10-slim

ENV PYTHONUNBUFFERED=True \
    PORT=9090

WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 _wsgi:app
