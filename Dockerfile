# базовый образ
FROM python:latest

RUN mkdir -p /usr/src/app/
WORKDIR /usr/src/app/

COPY . /usr/src/app/

RUN apt-get update && apt-get install -y sudo
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
# команда для запуска приложения
CMD ["python", "app.py"]

# docker build -t web-stegosystem .
# docker run --rm --name web -p 5000:5000 web-stegosystem