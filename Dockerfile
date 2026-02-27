FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8050

# --timeout 120: gives Pearl's LLM+Pinecone calls time to complete
# --workers 2: handles a few concurrent users without over-provisioning
CMD gunicorn viz:server --bind 0.0.0.0:${PORT:-8050} --workers 2 --timeout 120
