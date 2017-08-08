from flask import request, jsonify, Flask
import json
import classifier
import numpy as np
import tensorflow as tf
import requests
import cifar10filter as filter
import matplotlib.pyplot as plt
from PIL import Image
import variableextraction as variables
from werkzeug.datastructures import FileStorage

# Image path
image_path = "C:\\Users\\emman\\Desktop\\tensorflow\\TensorFlowClient\\images\\test7.jpg"

def send_processed_image():
    image = tf.gfile.FastGFile(image_path, 'rb').read()
    image_data = tf.cast(tf.image.decode_jpeg(image, channels=3), tf.float32)
    plt.savefig('C:\\Users\\emman\\Downloads\\test2.jpg')
    #filtered_image = filter.filterImage([image_data])
    #np_array_image = np.asarray(filtered_image)
    filtered_image = filter.filterImage(tf.expand_dims(image_data, 0))
    variables.getVariables()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        np_array_image = sess.run(filtered_image)

    print(np.asarray(np_array_image).ndim)
    print(frontend(np.asarray(np_array_image)).ndim)
    plt.imshow(frontend(np.asarray(np_array_image)))
    plt.savefig('C:\\Users\\emman\\Downloads\\test.jpg')

   # imgplot = plt.imshow(np.asarray(filtered_image))
   # json_data = json.loads(str(classify, encoding='utf-8'))
   # resp = requests.post("http://127.0.0.1:5000/tensorflow/", json={'testing': 'test', 'test2': 'test2'})
    headers = {"Content-Type": "multipart/form-data"}
    resp = requests.post("http://127.0.0.1:5000/tensorflow/", headers=headers, data={'image': open(image_path, 'rb')})
    print(resp)
  #  json_data = json.loads(str(resp.content, encoding='utf-8'))
    print(np_array_image)


def frontend(data):
    if data.ndim == 4:
        data = data.reshape([data.shape[0]*data.shape[1]*data.shape[2],data.shape[3]])
        return data
    elif data.ndim == 3:
        data = data.reshape([data.shape[0]*data.shape[1],data.shape[2]])
        return data
    elif data.ndim == 2:
        return data
    else:
        text = 'unrecognized shape'
        print(str(text))