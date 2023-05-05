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
conn.execute('CREATE TABLE IF NOT EXISTS images (filename VARCHAR(255) PRIMARY KEY, binary TEXT, phash VARCHAR(64))')
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
    data = {}
    for row in rows:
        data.update({row[0]: [row[1], row[2]]})
    data = json.dumps(data)

    return jsonify(data), 200


def get_ascii_symbols() -> list[str]:
    return [chr(i) for i in range(32, 128)]


@app.route('/api/data', methods=['POST'])
def post_data():
    data = request.get_json()
    conn = sqlite3.connect('./repository/images.db')
    c = conn.cursor()
    c.executemany('INSERT INTO images VALUES (?, ?, ?)', [(k, v[0], v[1]) for k, v in data.items()])
    conn.commit()
    conn.close()
    return 'OK', 201


@app.route('/')
def sender():
    key = int(os.getenv('KEY'))
    np.random.seed(key)

    # Путь к папке с изображениями
    path = 'resources/sent_images'

    # Создаем словарь для хранения и последующей передачи изображений в бинарном режиме
    images = {}
    # Создаем словарь для сохранения pHash
    hashes = []
    # Проходимся по всем файлам в папке
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        # Открываем изображение и вычисляем перцептивный хэш
        with Image.open(os.path.join(path, filename)) as image:
            phash = improved_phash(image)

        hashes.append(phash)
        binary_phash = np.binary_repr(phash, width=64)
        images.update({filename: [encoded_string, binary_phash]})

    requests.post(f'{request.host_url}/api/data', json=images, headers=headers)
    np.random.seed()
    return 'OK', 200


@app.route('/get_images')
def receiver():
    key = int(os.getenv('KEY'))
    np.random.seed(key)

    # Путь к папке, куда будет сохранять изображения
    path = 'resources/received_images'

    data_json = requests.get(f'{request.host_url}/api/data', headers=headers).json()
    # Convert the JSON string to a dictionary
    data = json.loads(data_json)
    # декодирование строки из формата base64
    for filename, bin_hash in data.items():
        decoded_string = base64.b64decode(bin_hash[0])
        # сохранение изображения в файл
        with open(os.path.join(path, filename), "wb") as image_file:
            image_file.write(decoded_string)

        # Открываем изображение
        with Image.open(os.path.join(path, filename)) as image:
            phash = improved_phash(image)

    np.random.seed()

    return 'OK', 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
