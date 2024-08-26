FROM --platform=linux/amd64 python:3.10.14-slim

WORKDIR /app

# Dependencies
COPY ./requirements.txt .
COPY ./main.py /app/main.py
RUN pip install --no-cache-dir -r requirements.txt

# Set entrypoint
ENTRYPOINT ["python3", "-m", "main"]