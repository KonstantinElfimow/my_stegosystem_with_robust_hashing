FROM python:latest

RUN mkdir -p /usr/src/app/
WORKDIR /usr/src/app/

COPY . /usr/src/app/

RUN apt-get update && apt-get install -y sudo
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]

# docker build -t web-stegosystem .
# docker run --rm --name web-app -p 5000:5000 -it -v web:/usr/src/app/repository web-stegosystem