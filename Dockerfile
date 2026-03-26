FROM python:3.10-slim

ARG RUN_ID

WORKDIR /app

RUN pip install mlflow

# Simulate downloading model
RUN echo "Downloading model for RUN_ID=${RUN_ID}"

CMD ["echo", "Model container ready"]