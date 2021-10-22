FROM python:3.8

RUN apt-get update && apt-get install -y

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000

ENTRYPOINT ["python","app.py"]