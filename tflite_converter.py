import os
import tensorflow as tf
import numpy as np


img_size = 240


def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 3, img_size, img_size)
      yield [data.astype(np.float32)]


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

converter = tf.lite.TFLiteConverter.from_saved_model(f"weights/test2.pb")

# int8 - Convert using CPU (not working in GPU - document here: https://www.tensorflow.org/lite/performance/post_training_quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8

# float16 - CPU and GPU are both working
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open(f"{str(img_size)}-int8.tflite", "wb").write(tflite_model)

# ISSUES
# AddV2 custom ops issue: https://github.com/tensorflow/tensorflow/issues/31901#issuecomment-607198654
# Solution: Add SELECT_TF_OPS to supported_ops

# Regular TensorFlow ops are not supported by this interpreter. Make sure you apply/link the Flex delegate before
# inference.Node number 1488 (FlexAddV2) failed to prepare.
# Reason: Using flex delegate in python is not yet supported (tflite_runtime - not sure it's supported now)
# Solution: Temporarily use tensorflow.lite as tflite rather than tflite_runtime.interpreter as tflite
# https://github.com/tensorflow/tensorflow/issues/40157#issuecomment-639244261
