# Use an official Python runtime as a base image
FROM python:3.9-slim

WORKDIR /backend

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5100
CMD ["python", "app.py"]