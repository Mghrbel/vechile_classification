
FROM python:3.8.0-slim

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 5000
COPY . .

ENTRYPOINT ["python", "app.py"]