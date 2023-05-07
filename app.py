import base64
import io
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

conn = sqlite3.connect('./repository/images.db')
# Дропаем, если таблица уже существует
conn.execute('DROP TABLE IF EXISTS images')
# Создаем таблицу для хранения изображений
conn.execute('CREATE TABLE IF NOT EXISTS images '
             '(filename VARCHAR(255) PRIMARY KEY, '
             'binary TEXT NOT NULL)')
conn.commit()
conn.close()

conn = sqlite3.connect('./repository/images.db')
# Дропаем, если таблица уже существует
conn.execute('DROP TABLE IF EXISTS cutouts')
# Создаем таблицу для хранения фрагментов изображений
conn.execute('CREATE TABLE IF NOT EXISTS cutouts '
             '(filename VARCHAR(255), '
             'start_i INTEGER, '
             'start_j INTEGER, '
             'end_i INTEGER, '
             'end_j INTEGER, '
             'PRIMARY KEY (filename, start_i, start_j, end_i, end_j))')
conn.commit()
conn.close()

load_dotenv('.env')


@app.route('/api/data/images', methods=['GET'])
def get_images():
    conn = sqlite3.connect('repository/images.db')
    c = conn.cursor()
    c.execute('SELECT * '
              'FROM images')
    rows = c.fetchall()
    conn.close()

    # Преобразуем данные в json
    data = {}
    for row in rows:
        d = {row[0]: row[1]}
        data.update(d)
    data = json.dumps(data)

    return jsonify(data), 200


@app.route('/api/data/images', methods=['POST'])
def post_images():
    data = request.get_json()
    conn = sqlite3.connect('repository/images.db')
    c = conn.cursor()
    unique_filenames = set()
    for k, v in data.items():
        if k not in unique_filenames:
            unique_filenames.add(k)
            c.execute('INSERT INTO images (filename, binary) VALUES (?, ?)', (k, v))
    conn.commit()
    conn.close()
    return 'OK', 201


@app.route('/api/data/cutouts', methods=['GET'])
def get_cutouts():
    conn = sqlite3.connect('repository/images.db')
    c = conn.cursor()
    c.execute('SELECT * '
              'FROM cutouts '
              'ORDER BY filename')
    rows = c.fetchall()
    conn.close()

    # Преобразуем данные в json
    data = []
    for row in rows:
        d = [row[0], row[1], row[2], row[3], row[4]]
        data.append(d)
    data = json.dumps(data)

    return jsonify(data), 200


@app.route('/api/data/cutouts', methods=['POST'])
def post_cutouts():
    data = request.get_json()
    conn = sqlite3.connect('repository/images.db')
    c = conn.cursor()
    c.executemany('INSERT INTO cutouts VALUES (?, ?, ?)', [(values[0], values[1], values[2]) for values in data])
    conn.commit()
    conn.close()
    return 'OK', 201


def gamma_function(start: int, stop: int):
    return np.random.randint(start, stop)


@app.route('/')
def sender():
    # Путь к папке с изображениями
    path = 'repository/resources/sent_images'

    # Создаем список для хранения и последующей передачи изображений в бинарном режиме
    filename_binary = {}
    # Проходимся по всем файлам в папке
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        filename_binary[filename] = encoded_string
    # Через REST API отправляем изображения в виде бинарника
    requests.post(f'{request.host_url}/api/data/images', json=filename_binary, headers=headers)

    # Создаем словарь для сохранения pHash
    hashes = {}
    # Проходимся по всем файлам в папке
    for filename in os.listdir(path):
        # Открываем изображение и вычисляем перцептивный хэш
        with Image.open(os.path.join(path, filename)) as image:
            # Получаем размер изображения
            width, height = image.size

            step = 32
            for i in range(step, height, 32):
                for j in range(step, width, 32):
                    cropped_image = image.crop((i - step, j - step, i, j))
                    phash: np.uint8 = improved_phash(cropped_image)

                    start = tuple([i - step, j - step])
                    end = tuple([i, j])
                    hashes.setdefault(phash, []).append([filename, start, end])

    print(len(hashes))
    # print(hashes)

    path = 'repository/resources/message.txt'
    with open(path, mode='r', encoding='utf-8') as file:
        message = file.read()

    key = int(os.getenv('KEY'))

    np.random.seed(key)
    chosen_fragments = []
    for ch in list(message):
        ch_ord = np.uint8(int(''.join(format(x, '08b') for x in ch.encode('utf-8')), 2))
        gamma = np.uint8(gamma_function(0, 2 ** 8))

        result: np.uint8 = np.bitwise_xor(ch_ord, gamma)
        if hashes.get(result, None) is not None:
            chosen_fragments.append(hashes[result][0])
        else:
            print(chosen_fragments)
            raise ValueError('Map не был полностью заполнен!')
    np.random.seed()

    requests.post(f'{request.host_url}/api/data/cutouts', json=chosen_fragments, headers=headers)

    return 'OK', 200


@app.route('/get_images')
def receiver():
    filename = int(os.getenv('KEY'))
    np.random.seed(filename)

    # Путь к папке, куда будет сохранять изображения
    path = 'repository/resources/received_images'

    data_json = requests.get(f'{request.host_url}/api/data/images', headers=headers).json()
    # Преобразуем json в dict
    data = json.loads(data_json)
    # декодирование строки из формата base64
    for filename, binary in data.items():
        # сохранение изображения в файл
        with open(os.path.join(path, filename), 'wb') as image_file:
            decoded_string = base64.b64decode(binary)
            image_file.write(decoded_string)

    # Храним полученные хэши
    hashes = []
    # Открываем изображение и вычисляем перцептивный хэш
    data_json = requests.get(f'{request.host_url}/api/data/cutouts', headers=headers).json()
    # Преобразуем json
    data = json.loads(data_json)
    for values in data:
        filename = values[0]
        with Image.open(filename) as image:
            start_i = int(values[1])
            start_j = int(values[2])
            end_i = int(values[3])
            end_j = int(values[4])

            cropped_image = image.crop((start_i, start_j, end_i, end_j))
        phash: np.uint8 = improved_phash(cropped_image)
        hashes.append(phash)

    # Декодируем сообщение
    buffer = io.StringIO()
    for h in hashes:
        gamma = gamma_function(0, 2 ** 8)
        m: np.uint8 = np.bitwise_xor(h, gamma)

        m_i = (int(m).to_bytes(1, byteorder='big')).decode('utf-8')
        buffer.write(m_i)

    with open(os.path.join(path, 'message_recovered.txt'), mode='w', encoding='utf-8') as file:
        file.write(buffer.getvalue())

    np.random.seed()
    return 'OK', 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
