FROM python
WORKDIR /app
COPY requirements.txt .
COPY README.md .
COPY src/ ./src/

CMD ["python", "src/docker.py"]