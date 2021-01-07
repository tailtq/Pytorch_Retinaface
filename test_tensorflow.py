import os
import cv2
import numpy as np
import tensorflow as tf
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.list_physical_devices('GPU')
print("Devices: ", gpus)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, False)

img = cv2.imread("imgs/test-img.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (480, 480)).transpose(2, 0, 1)
imgs = np.expand_dims(img, axis=0)

with tf.device("/gpu:0"):
    imported = tf.saved_model.load('tf-4802')
    inference_func = imported.signatures["serving_default"]

    imgs = tf.convert_to_tensor(imgs, dtype=tf.float32)

    for i in range(100):
        start_time = time.time()
        inference_func(input=imgs)
        print(time.time() - start_time)

# print(tf.TensorSpec((1, 3, 480, 480), dtype=tf.float32))
#
# # result = imported(tf.convert_to_tensor(img, dtype=tf.float32))
# result = imported(input=tf.TensorSpec((1, 3, 480, 480), dtype=tf.float32, name="input"))
# print(result)
