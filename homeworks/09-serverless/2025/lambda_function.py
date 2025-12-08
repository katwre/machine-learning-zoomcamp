import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO
from urllib import request

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    return img

url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"

img = download_image(url)
img = prepare_image(img, (200, 200))

x = np.array(img, dtype=np.float32)
x = x / 255.0

x = np.transpose(x, (2, 0, 1))
x = np.expand_dims(x, axis=0)

#x = np.transpose(x, (0, 3, 1, 2))

session = ort.InferenceSession("hair_classifier_empty.onnx")

pred = session.run(["output"], {"input": x})

print(pred)
