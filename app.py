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

# Подключаемся к базе данных
conn = sqlite3.connect('./repository/images.db')
# Дропаем, если db уже существует
conn.execute('DROP TABLE IF EXISTS images')
# Создаем таблицу для хранения изображений
conn.execute('CREATE TABLE IF NOT EXISTS images '
             '(id INTEGER PRIMARY KEY AUTOINCREMENT,'
             'filename VARCHAR(255) NOT NULL, '
             'binary TEXT NOT NULL)')
# Сохраняем изменения в базе данных
conn.commit()
# Закрываем соединение с базой данных
conn.close()


load_dotenv('.env')


@app.route('/api/data', methods=['GET'])
def get_data():
    conn = sqlite3.connect('repository/images.db')
    c = conn.cursor()
    c.execute('SELECT * FROM images')
    rows = c.fetchall()
    conn.close()

    # Преобразуем данные в json
    data = []
    for row in rows:
        d = {row[0]: [row[1], row[2]]}
        data.append(d)
    data = json.dumps(data)

    return jsonify(data), 200


@app.route('/api/data', methods=['POST'])
def post_data():
    data = request.get_json()
    conn = sqlite3.connect('repository/images.db')
    c = conn.cursor()
    c.executemany('INSERT INTO images (filename, binary, phash) VALUES (?, ?)', [(row[0], row[1]) for row in data])
    conn.commit()
    conn.close()
    return 'OK', 201


def gamma_function(start: int, stop: int):
    return np.random.randint(start, stop)


@app.route('/')
def sender():
    key = int(os.getenv('KEY'))
    np.random.seed(key)

    # Путь к папке с изображениями
    path = 'repository/resources/sent_images'
    # Создаем словарь для сохранения pHash
    hashes = {}
    # Проходимся по всем файлам в папке
    for filename in os.listdir(path):
        # Открываем изображение и вычисляем перцептивный хэш
        with Image.open(os.path.join(path, filename)) as image:
            phash: np.uint16 = improved_phash(image)
        hashes[phash] = hashes.setdefault(phash, []).append(filename)

    path = 'repository/resources/message.txt'
    with open(path, mode='r', encoding='utf-8') as file:
        message = file.read()

    chosen_files = []
    for ch in list(message):
        ch_ord = np.uint16(int(''.join(format(x, '08b') for x in ch.encode('utf-8')), 2))
        gamma = np.uint16(gamma_function(0, 2 ** 16))

        result: np.uint16 = np.bitwise_xor(ch_ord, gamma)
        if hashes.get(result, None) is not None:
            chosen_files.append(hashes[result][0])
        else:
            raise ValueError('Map не был полностью заполнен!')
        
    # Создаем список для хранения и последующей передачи изображений в бинарном режиме
    filename_binary = []
    # Проходимся по всем файлам в папке
    for filename in chosen_files:
        with open(os.path.join(path, filename), 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        filename_binary.append([filename, encoded_string])
    
    # Через REST API отправляем изображения в виде бинарника
    requests.post(f'{request.host_url}/api/data', json=filename_binary, headers=headers)

    np.random.seed()
    return 'OK', 200


@app.route('/get_images')
def receiver():
    key = int(os.getenv('KEY'))
    np.random.seed(key)

    # Путь к папке, куда будет сохранять изображения
    path = 'repository/resources/received_images'

    # Храним полученные хэши
    hashes = []

    data_json = requests.get(f'{request.host_url}/api/data', headers=headers).json()
    # Преобразуем json в dict
    data = json.loads(data_json)
    # декодирование строки из формата base64
    for row in data:
        for key, values in row.items():
            # сохранение изображения в файл
            filename = values[0]
            with open(os.path.join(path, filename), 'wb') as image_file:
                encoded_string = values[1]
                decoded_string = base64.b64decode(encoded_string)
                image_file.write(decoded_string)

            # Открываем изображение и вычисляем перцептивный хэш
            with Image.open(os.path.join(path, filename)) as image:
                phash: np.uint16 = improved_phash(image)
            hashes.append(phash)

    # Декодируем сообщение
    buffer = io.StringIO()
    for h in hashes:
        gamma = gamma_function(0, 2 ** 16)
        m: np.uint16 = np.bitwise_xor(h, gamma)

        m_i = (int(m).to_bytes(2, byteorder='big')).decode('utf-8')
        buffer.write(m_i)

    with open(os.path.join(path, 'message_recovered.txt'), mode='w', encoding='utf-8') as file:
        file.write(buffer.getvalue())

    np.random.seed()
    return 'OK', 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
