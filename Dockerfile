FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8050

# --workers 1: _stream_buffers is in-process memory — multiple workers would
# split requests across isolated processes, breaking the streaming buffer.
# --threads 4: handles concurrent users within the single process instead.
CMD gunicorn viz:server --bind 0.0.0.0:${PORT:-8050} --workers 1 --threads 4 --timeout 120
