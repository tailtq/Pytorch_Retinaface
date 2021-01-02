import os
import time
import argparse

import numpy as np
from PIL import Image
# import tflite_runtime.interpreter as tflite
import tensorflow.lite as tflite


# def load_labels(filename):
#     with open(filename, 'r') as f:
#         return [line.strip() for line in f.readlines()]


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        default='/tmp/grace_hopper.bmp',
        help='image to be classified')
    parser.add_argument(
        '-m',
        '--model_file',
        default='/tmp/mobilenet_v1_1.0_224_quant.tflite',
        help='.tflite model to be executed')
    parser.add_argument(
        '-l',
        '--label_file',
        default='/tmp/labels.txt',
        help='name of file containing labels')
    parser.add_argument(
        '--input_mean',
        default=127.5, type=float,
        help='input_mean')
    parser.add_argument(
        '--input_std',
        default=127.5, type=float,
        help='input standard deviation')
    parser.add_argument(
        '--num_threads', default=None, type=int, help='number of threads')
    args = parser.parse_args()

    interpreter = tflite.Interpreter(model_path=args.model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    input_type = input_details[0]['dtype']
    print('input: ', input_type)
    output_type = output_details[0]['dtype']
    print('output: ', output_type)

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][2]
    width = input_details[0]['shape'][3]
    img = Image.open(args.image).resize((width, height))

    # add N dim
    input_data = np.expand_dims(img, axis=0).transpose((0, 3, 1, 2))
    print(input_data.shape)

    # Cannot use GPU delegate: https://www.tensorflow.org/lite/performance/delegates (work for Android + iOS only based on document)

    if floating_model:
        input_data = (np.float32(input_data) - args.input_mean) / args.input_std

    for i in range(100):
        interpreter.set_tensor(input_details[0]['index'], input_data)

        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()

        print(interpreter.get_tensor(output_details[0]['index']).shape)
        print(interpreter.get_tensor(output_details[1]['index']).shape)
        print(interpreter.get_tensor(output_details[2]['index']).shape)

        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)

        print('time: {:.4f}'.format((stop_time - start_time)))
