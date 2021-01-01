import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("test2")
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.inference_type = tf.uint8
tflite_model = converter.convert()
open("test2.tflite", "wb").write(tflite_model)

# ISSUES
# AddV2 custom ops issue: https://github.com/tensorflow/tensorflow/issues/31901#issuecomment-607198654
# Solution: Add SELECT_TF_OPS to supported_ops

# Regular TensorFlow ops are not supported by this interpreter. Make sure you apply/link the Flex delegate before
# inference.Node number 1488 (FlexAddV2) failed to prepare.
# Reason: Using flex delegate in python is not yet supported
# https://github.com/tensorflow/tensorflow/issues/40157#issuecomment-639244261
