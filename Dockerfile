# базовый образ
FROM python:3.10

# установка зависимостей
RUN pip install flask

# копирование файлов приложения
COPY app.py /app/

# рабочая директория
WORKDIR /app

# команда для запуска приложения
CMD ["python", "app.py"]