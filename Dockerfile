FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/MultiNEAT
COPY . .
RUN pip install --no-cache-dir ".[test]" && python -m pytest tests -q

CMD ["python", "examples/TestNEAT_xor.py"]
