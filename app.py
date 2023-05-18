import base64
import io
import json
import os
import time
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import requests
import sqlite3
from dotenv import load_dotenv
from flask import Flask, jsonify, request
import robust_hashing as rh

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


class MyConstants:
    HASH_SIZE: int = 16
    HIGHFREQ_FACTOR: int = 4
    WINDOW_SIZE: int = 128


def format_time(seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return '{:d} дней, {:02d}:{:02d}:{:.2f}'.format(int(days), int(hours), int(minutes), seconds)


def gamma_function(stop: int) -> np.uint:
    return np.random.randint(0, stop)


@app.route('/')
def sender():
    logger = io.StringIO()

    images_dir: str = 'repository/resources/sent_images'
    message_path: str = 'repository/resources/message.txt'
    logger_path: str = 'sender_log.txt'

    # Начало заполнения map
    start_time = time.perf_counter()

    # Создаем словарь для сохранения перцептивных хешей и их кадров {pHash: [cutout_1, ...]}
    hashes: dict[int: list] = dict()
    # Создаём counter для отслеживания коллизий в map
    counter = 0
    # Проходимся по всем файлам в папке
    for filename in os.listdir(images_dir):
        # Открываем изображение и вычисляем перцептивный хэш
        with Image.open(os.path.join(images_dir, filename)) as image:
            arr = np.asarray(image, dtype=np.uint8)

        block_size = MyConstants.WINDOW_SIZE

        height, width = arr.shape[0:2]
        for i in range(0, height - block_size + 1, block_size):
            for j in range(0, width - block_size + 1, block_size):
                block = arr[i:i + block_size, j:j + block_size]

                h: rh.ImageHash = rh.phash(Image.fromarray(block),
                                           hash_size=MyConstants.HASH_SIZE,
                                           highfreq_factor=MyConstants.HIGHFREQ_FACTOR)

                left = j
                upper = i
                right = left + block_size
                lower = upper + block_size
                coordinates: tuple = left, upper, right, lower

                hashes.setdefault(h.value, []).append([filename, coordinates])
                counter += 1
    # Окончание заполнения map
    end_time = time.perf_counter()
    # Примерное время заполнения map всевозможными ключами
    approximate_time = format_time(((end_time - start_time) * (2 ** MyConstants.HASH_SIZE)) / len(hashes))

    logger.write('Примерное время заполнения всего map:\n{}\n'.format(approximate_time))
    logger.write('Map заполнен на {} из {}\n'.format(len(hashes), 2 ** MyConstants.HASH_SIZE))
    logger.write('Коллизий в map: {}\n'.format(counter - len(hashes)))
    del counter

    try:
        with open(message_path, mode='r', encoding='utf-8') as file:
            message = file.read()

        key = int(os.getenv('KEY'))
        np.random.seed(key)

        chosen_frames = []

        for ch in message:
            gamma = gamma_function(2 ** MyConstants.HASH_SIZE)
            ch_ord = np.uint8(int(''.join(format(x, '08b') for x in ch.encode('utf-8')), 2))
            result = np.bitwise_xor(ch_ord, gamma)

            if hashes.get(result, None) is None:
                raise ValueError('Map не был полностью заполнен!')

            chosen_frames.append(hashes[result][0])
    except ValueError as e:
        logger.write(f'Ошибка: {str(e)}')
        raise e
    else:
        logger.write('Сообщение было успешно закодировано!\n')
    finally:
        with open(logger_path, mode='w', encoding='utf-8') as file:
            file.write(logger.getvalue())
        del logger
        np.random.seed()

    # Создаем словарь для хранения и последующей передачи изображений в бинарном режиме
    filename_binary = {}
    counter = 0
    for filename_coors in chosen_frames:
        filename: str = filename_coors[0]
        coordinates: tuple = filename_coors[1]

        suffix = filename.split('.')[-1]

        with Image.open(os.path.join(images_dir, filename)) as image:
            frame = image.crop(coordinates)

        byte_stream = io.BytesIO()
        f = 'JPEG' if suffix.lower() == 'jpg' else suffix
        frame.save(byte_stream, format=f)
        byte_stream.seek(0)

        encoded_string = base64.b64encode(byte_stream.read()).decode('utf-8')
        filename_binary[f'{counter}.{suffix}'] = encoded_string
        counter += 1
    # Через REST API отправляем изображения в виде бинарника
    requests.post(f'{request.host_url}/api/data/images', json=filename_binary, headers=headers)

    return 'OK', 200


@app.route('/get_images')
def receiver():
    logger = io.StringIO()

    images_dir: str = 'repository/resources/received_images'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    message_path: str = 'repository/resources/message_recovered.txt'
    logger_path: str = 'receiver_log.txt'

    # Храним полученные хэши
    hashes: list[int] = []

    data_json = requests.get(f'{request.host_url}/api/data/images', headers=headers).json()
    # Преобразуем json в dict
    data = json.loads(data_json)
    # декодирование строки из формата base64
    for filename, binary in data.items():
        # сохранение изображения в файл
        with open(os.path.join(images_dir, filename), mode='wb') as image_file:
            decoded_string = base64.b64decode(binary)
            image_file.write(decoded_string)

        with Image.open(os.path.join(images_dir, filename)) as image:
            h: rh.ImageHash = rh.phash(image,
                                       hash_size=MyConstants.HASH_SIZE,
                                       highfreq_factor=MyConstants.HIGHFREQ_FACTOR)
            hashes.append(h.value)

    logger.write('Изображения были получены!\n')

    # Декодируем сообщение
    buffer = io.StringIO()

    key = int(os.getenv('KEY'))
    np.random.seed(key)

    for h in hashes:
        gamma = gamma_function(2 ** MyConstants.HASH_SIZE)
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


def test_phash():
    filename = 'test.png'

    with Image.open(filename) as image:
        h = rh.phash(image,
                     hash_size=MyConstants.HASH_SIZE,
                     highfreq_factor=MyConstants.HIGHFREQ_FACTOR)
    for r in range(1, 30, 5):
        with Image.open(filename) as image:
            rot_h = rh.phash(image.rotate(r),
                             hash_size=MyConstants.HASH_SIZE,
                             highfreq_factor=MyConstants.HIGHFREQ_FACTOR)

        print('Поворот (градус: {}): {} расстояние Хемминга'.format(r, h - rot_h))

    # Преобразование изображения в массив NumPy
    with Image.open(filename) as image:
        image_array = np.array(image)

    # Создание гауссовского шума
    noise = np.random.normal(0, 10, image_array.shape)
    noisy_image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)

    # Преобразование массива NumPy обратно в изображение
    noisy_image = Image.fromarray(noisy_image_array)
    for r in range(1, 30, 5):
        gauss_h = rh.phash(noisy_image.filter(ImageFilter.GaussianBlur(radius=r)),
                           hash_size=MyConstants.HASH_SIZE,
                           highfreq_factor=MyConstants.HIGHFREQ_FACTOR)

        print('Гауссовский шум (радиус: {}): {} расстояние Хемминга'.format(r, h - gauss_h))

    # Изменение яркости изображения (затемнение)
    for x in range(1, 10):
        with Image.open(filename) as image:
            enhancer = ImageEnhance.Brightness(image).enhance(x / 10)
        brightened_h = rh.phash(enhancer,
                                hash_size=MyConstants.HASH_SIZE,
                                highfreq_factor=MyConstants.HIGHFREQ_FACTOR)
        print('Изменение яркости изображения (на {}%): {} расстояние Хемминга'.format(100 - x * 10, h - brightened_h))


if __name__ == '__main__':
    # test_phash()
    app.run(host='0.0.0.0', port=port, debug=True)
