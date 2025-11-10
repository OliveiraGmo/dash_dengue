FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential && \
	rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.13-slim

WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv
COPY ./src ./src
COPY ./README.md ./

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
