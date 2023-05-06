import base64
import json
import os
from PIL import Image
import numpy as np
import requests
import sqlite3
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from my_modules.robust_hashing import improved_phash


app = Flask(__name__)

port = 5000
headers = {'Content-Type': 'application/json'}

# Подключаемся к базе данных
conn = sqlite3.connect('./repository/images.db')
# Дропаем, если db уже существует
conn.execute('DROP TABLE IF EXISTS images')
# Создаем таблицу для хранения изображений
conn.execute('CREATE TABLE IF NOT EXISTS images '
             '(id INTEGER PRIMARY KEY AUTOINCREMENT,'
             'filename VARCHAR(255) NOT NULL, '
             'binary TEXT NOT NULL, '
             'phash VARCHAR(64) NOT NULL)')
# Сохраняем изменения в базе данных
conn.commit()
# Закрываем соединение с базой данных
conn.close()


load_dotenv('.env')


@app.route('/api/data', methods=['GET'])
def get_data():
    conn = sqlite3.connect('./repository/images.db')
    c = conn.cursor()
    c.execute('SELECT * FROM images')
    rows = c.fetchall()
    conn.close()

    # Преобразуем данные в json
    data = []
    for row in rows:
        d = {row[0]: [row[1], row[2], row[3]]}
        data.append(d)
    data = json.dumps(data)

    return jsonify(data), 200


def get_ascii_symbols() -> list[str]:
    return [chr(i) for i in range(32, 128)]


@app.route('/api/data', methods=['POST'])
def post_data():
    data = request.get_json()
    conn = sqlite3.connect('./repository/images.db')
    c = conn.cursor()
    c.executemany('INSERT INTO images (filename, binary, phash) VALUES (?, ?, ?)',
                  [(row[0], row[1], row[2]) for row in data])
    conn.commit()
    conn.close()
    return 'OK', 201


@app.route('/')
def sender():
    key = int(os.getenv('KEY'))
    np.random.seed(key)

    # Путь к папке с изображениями
    path = 'repository/resources/sent_images'

    # Создаем список для хранения и последующей передачи изображений в бинарном режиме
    images = []
    # Создаем словарь для сохранения pHash
    hashes = []
    # Проходимся по всем файлам в папке
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        # Открываем изображение и вычисляем перцептивный хэш
        with Image.open(os.path.join(path, filename)) as image:
            phash: np.uint64 = improved_phash(image)

        hashes.append(phash)
        binary_phash = np.binary_repr(phash, width=64)
        images.append([filename, encoded_string, binary_phash])

    requests.post(f'{request.host_url}/api/data', json=images, headers=headers)
    np.random.seed()
    return 'OK', 200


@app.route('/get_images')
def receiver():
    key = int(os.getenv('KEY'))
    np.random.seed(key)

    # Путь к папке, куда будет сохранять изображения
    path = 'repository/resources/received_images'

    data_json = requests.get(f'{request.host_url}/api/data', headers=headers).json()
    # Convert the JSON string to a dictionary
    data = json.loads(data_json)
    # декодирование строки из формата base64
    for row in data:
        for key, values in row.items():
            # сохранение изображения в файл
            filename = values[0]
            with open(os.path.join(path, filename), "wb") as image_file:
                encoded_string = values[1]
                decoded_string = base64.b64decode(encoded_string)
                image_file.write(decoded_string)

            # Открываем изображение и вычисляем перцептивный хэш
            original_phash = np.uint64(int(values[2], 2))
            with Image.open(os.path.join(path, filename)) as image:
                phash: np.uint64 = improved_phash(image)

            if phash == original_phash:
                print(f'{filename} изображение оригинальное!')
            else:
                print('Изображения различны!')

    np.random.seed()
    return 'OK', 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
