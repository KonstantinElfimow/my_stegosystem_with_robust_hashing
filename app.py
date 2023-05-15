import base64
import io
import json
import os
import time
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
             '(id INTEGER PRIMARY KEY AUTOINCREMENT, '
             'filename VARCHAR(255), '
             'start_i INTEGER, '
             'start_j INTEGER, '
             'end_i INTEGER, '
             'end_j INTEGER)')
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
    data = json.dumps(data, ensure_ascii=False)

    return jsonify(data), 200


@app.route('/api/data/images', methods=['POST'])
def post_images():
    data = request.get_json()
    conn = sqlite3.connect('repository/images.db')
    c = conn.cursor()
    c.executemany('INSERT OR REPLACE INTO images (filename, binary) VALUES (?, ?)', [(k, v) for k, v in data.items()])
    conn.commit()
    conn.close()
    return 'OK', 201


@app.route('/api/data/cutouts', methods=['GET'])
def get_cutouts():
    conn = sqlite3.connect('repository/images.db')
    c = conn.cursor()
    c.execute('SELECT * '
              'FROM cutouts')
    rows = c.fetchall()
    conn.close()

    # Преобразуем данные в json
    data = {}
    for row in rows:
        d = {row[0]: [row[1], row[2], row[3], row[4], row[5]]}
        data.update(d)
    data = json.dumps(data, ensure_ascii=False)

    return jsonify(data), 200


@app.route('/api/data/cutouts', methods=['POST'])
def post_cutouts():
    data = request.get_json()
    conn = sqlite3.connect('repository/images.db')
    c = conn.cursor()
    c.executemany('INSERT OR REPLACE INTO cutouts (filename, start_i, start_j, end_i, end_j) VALUES (?, ?, ?, ?, ?)',
                  [(l[0], l[1], l[2], l[3], l[4]) for l in data])
    conn.commit()
    conn.close()
    return 'OK', 201


def format_time(seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return '{:d} дней, {:02d}:{:02d}:{:.2f}'.format(int(days), int(hours), int(minutes), seconds)


def gamma_function(start: int, stop: int):
    return np.random.randint(start, stop)


class MyConstants:
    PHASH_SIZE = improved_phash.__annotations__['return'](0).dtype.itemsize * 8


@app.route('/')
def sender():
    logger = io.StringIO()

    images_dir: str = 'repository/resources/sent_images'
    message_path: str = 'repository/resources/message.txt'
    logger_path: str = 'sender_log.txt'

    # Создаем словарь для сохранения перцептивных хешей и их кадров {pHash: [cutout_1, ...]}
    hashes = {}
    # Создаём counter для отслеживания коллизий в map
    counter = 0

    # Начало заполнения map
    start_time = time.perf_counter()
    # Проходимся по всем файлам в папке
    for filename in os.listdir(images_dir):
        # Открываем изображение и вычисляем перцептивный хэш
        with Image.open(os.path.join(images_dir, filename)) as image:
            arr = np.asarray(image.convert('YCbCr'), dtype=np.uint8)
            # Получаем размер изображения
            height, width = arr.shape[0:2]
            # Размер окна
            step = 96
            for i in range(0, height - step + 1, step):
                for j in range(0, width - step + 1, step):
                    start_i = i
                    start_j = j
                    end_i = i + step
                    end_j = j + step
                    phash: np.uint8 = improved_phash(Image.fromarray(arr[start_i: end_i, start_j: end_j]))
                    hashes.setdefault(phash, []).append([filename, start_i, start_j, end_i, end_j])
                    counter += 1
    # Окончание заполнения map
    end_time = time.perf_counter()
    # Примерное время заполнения map всевозможными ключами
    approximate_time = format_time(((end_time - start_time) * (2 ** MyConstants.PHASH_SIZE)) / len(hashes))

    logger.write('Примерное время заполнения всего map:\n{}\n'.format(approximate_time))
    logger.write('Map заполнен на {} из {}\n'.format(len(hashes), 2 ** MyConstants.PHASH_SIZE))
    logger.write('Коллизий в map: {}\n'.format(counter - len(hashes)))

    try:
        with open(message_path, mode='r', encoding='utf-8') as file:
            message = file.read()

        key = int(os.getenv('KEY'))
        np.random.seed(key)

        chosen_fragments = []

        for ch in message:
            gamma = np.uint8(gamma_function(0, 2 ** MyConstants.PHASH_SIZE))

            ch_ord = np.uint8(int(''.join(format(x, '08b') for x in ch.encode('utf-8')), 2))

            result = np.uint8(np.bitwise_xor(ch_ord, gamma))

            if hashes.get(result, None) is None:
                raise ValueError('Map не был полностью заполнен!')

            chosen_fragments.append(hashes[result][0])
    except ValueError as e:
        logger.write(f'Ошибка: {str(e)}')
        raise e
    else:
        requests.post(f'{request.host_url}/api/data/cutouts', json=chosen_fragments, headers=headers)
        logger.write('Сообщение было успешно закодировано!\n')
    finally:
        np.random.seed()

    with open(logger_path, mode='w', encoding='utf-8') as file:
        file.write(logger.getvalue())
    del logger

    # Создаем список для хранения и последующей передачи изображений в бинарном режиме
    filename_binary = {}
    # Проходимся по всем файлам в папке
    for filename in os.listdir(images_dir):
        with open(os.path.join(images_dir, filename), 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        filename_binary[filename] = encoded_string
    # Через REST API отправляем изображения в виде бинарника
    requests.post(f'{request.host_url}/api/data/images', json=filename_binary, headers=headers)

    with open(logger_path, mode='a', encoding='utf-8') as file:
        file.write('Успех!')

    return 'OK', 200


@app.route('/get_images')
def receiver():
    logger = io.StringIO()

    images_dir: str = 'repository/resources/received_images'
    message_path: str = 'repository/resources/message_recovered.txt'
    logger_path: str = 'receiver_log.txt'

    data_json = requests.get(f'{request.host_url}/api/data/images', headers=headers).json()
    # Преобразуем json в dict
    data = json.loads(data_json)
    # декодирование строки из формата base64
    for filename, binary in data.items():
        # сохранение изображения в файл
        with open(os.path.join(images_dir, filename), mode='wb') as image_file:
            decoded_string = base64.b64decode(binary)
            image_file.write(decoded_string)
    logger.write('Изображения были получены!\n')

    # Храним полученные хэши
    hashes = []
    # Открываем изображение и вычисляем перцептивный хэш
    data_json = requests.get(f'{request.host_url}/api/data/cutouts', headers=headers).json()
    # Преобразуем json
    data = json.loads(data_json)
    for values in data.values():
        filename = values[0]
        start_i = int(values[1])
        start_j = int(values[2])
        end_i = int(values[3])
        end_j = int(values[4])
        with Image.open(os.path.join(images_dir, filename)) as image:
            arr = np.asarray(image.convert('YCbCr'), dtype=np.uint8)

        phash: np.uint8 = improved_phash(Image.fromarray(arr[start_i: end_i, start_j: end_j]))
        hashes.append(phash)

    # Декодируем сообщение
    buffer = io.StringIO()

    key = int(os.getenv('KEY'))
    np.random.seed(key)

    for h in hashes:
        gamma = np.uint8(gamma_function(0, 2 ** MyConstants.PHASH_SIZE))

        m = np.uint8(np.bitwise_xor(h, gamma))

        m_i = str(int(m).to_bytes(1, byteorder='little', signed=False), encoding='utf-8')
        buffer.write(m_i)

    np.random.seed()

    with open(message_path, mode='w', encoding='utf-8') as file:
        file.write(buffer.getvalue())
    del buffer

    logger.write('Сообщение успешно декодировано!')
    with open(logger_path, mode='w', encoding='utf-8') as file:
        file.write(logger.getvalue())

    return 'OK', 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
