FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OUTBOX_PATH=/data/matcher_outbox.sqlite3 \
    STATUS_PORT=8090

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
RUN useradd --system --no-create-home matcher && mkdir /data && chown matcher /data
USER matcher
# The outbox must survive restarts: mount a volume here.
VOLUME ["/data"]

EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"STATUS_PORT\"]}/status', timeout=4)"
CMD ["python", "-m", "app"]
