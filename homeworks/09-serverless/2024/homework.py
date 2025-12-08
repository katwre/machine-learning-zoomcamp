import numpy as np

import tflite_runtime.interpreter as tflite


from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


path2model = "/home/katwre/projects/machine-learning-zoomcamp/homeworks/09-serverless/model_2024_hairstyle.tflite"
#interpreter = tflite.Interpreter(model_path=path2model)
#interpreter.allocate_tensors()

#input_index = interpreter.get_input_details()[0]['index']
#output_index = interpreter.get_output_details()[0]['index']

#img=download_image("https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg")
#img = prepare_image(img, target_size=(200,200))


def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x

def predict(url, target_size):
    img = download_image(url)
    img = img.resize(target_size, Image.NEAREST)
    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = preprocess_input(X)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    preds = interpreter.get_tensor(output_index)

    return float(preds[0,0])
#url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
#predict(url, target_size=(200,200))

def lambda_handler(event, context):
    url = event['url']
    pred = predict(url, target_size=(200,200))
    return {
        'prediction': pred
    }

