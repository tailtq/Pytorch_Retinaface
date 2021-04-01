# Convert model from PyTorch to Tensorflow Lite

In this tutorial, we will convert model trained in PyTorch to Tensorflow Lite version. This approach attempts to run our models in mobile.

The PyTorch model I am using here is [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) with MobileNet backbone. The journey is not so simple, we have to convert PyTorch --> ONNX --> Tensorflow --> Tensorflow Lite. So in this article, I will write down the steps for revision and future projects.

## 1. Step 1 - Convert PyTorch to ONNX model

```python
from data import cfg_mnet
from models.retinaface import RetinaFace
from detect import load_model
import torch

net = RetinaFace(cfg=cfg_mnet, phase='test')
net = load_model(net, "weights/mobilenet0.25_Final.pth", load_to_cpu=False)

# test model again
# this code was copied from PyTorch official document on converting from PyTorch to ONNX
x = torch.randn(1, 3, 480, 480, requires_grad=True)
torch_out = net(x)
torch.onnx.export(net,  # model being run
                  x,  # model input (or a tuple for multiple inputs)
                  "weights/test2.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=["input"],  # the model's input names
                  output_names=["output"],  # the model's output names
                  dynamic_axes={"input": {0: "batch_size"},  # variable lenght axes
                                "output": {0: "batch_size"}})
```


## 2. Step 2 - Benchmark ONNX and PyTorch model

```python
import cv2
import torch
import onnxruntime
import numpy as np

from data import cfg_re50
from detect import load_model
from models.retinaface import RetinaFace


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


device = torch.device("cuda")

torch.set_grad_enabled(False)
net = RetinaFace(cfg=cfg_re50, phase='test')
net = load_model(net, "weights/mobilenet0.25_Final.pth", False)
net = net.to(device)
net.eval()

ort_session = onnxruntime.InferenceSession("weights/test2.onnx")

img = cv2.imread("./imgs/test-img2.jpeg", cv2.IMREAD_COLOR)
img = np.float32(img)
img -= (104, 117, 123)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (480, 480))
img = img.transpose((2, 0, 1))
img = torch.from_numpy(img).unsqueeze(0)
img = img.to(device)

# compute PyTorch prediction
loc, conf, landms = net(img)

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
loc2, conf2, landms2 = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(loc), loc2, rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(to_numpy(conf), conf2, rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(to_numpy(landms), landms2, rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
```


## 3. Step 3 - Convert ONNX to Tensorflow model

Pull docker image of tf-nightly: `docker pull tensorflow/tensorflow:devel-gpu`

Run container: `docker run -it --rm tensorflow/tensorflow:devel-gpu bash -v /path/to/current/dir:/app`

Install tf-nightly:
```shell
# Some layers are missing when using Tensorflow 1.15 (TF - TFLite).
# Also by using version 1.15, TF model getting from ONNX is automatically frozen (but it's not the case for new layer in TF2).
# So by some suggests, I switched to tf-nightly with CUDA 11.0 and CuDNN 8.0.
# Don't know why the heck should I do it because we're using tf-nightly version @@.
# But leverage its environment (CUDA, CuDNN is enough already)
pip install tf-nightly
pip install --upgrade git+https://github.com/onnx/onnx-tensorflow.git

cd /app && onnx-tf convert -i weights/test2.onnx -o weights/test2.pb
```

Now you can see your test2.pb folder in your root project. You can also test your converted model by using `test_tensorflow.py` file.


## 4. Step 4 - Convert Tensorflow to Tensorflow Lite

```python
import os
import tensorflow as tf
import numpy as np

def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 3, 480, 480)
      yield [data.astype(np.float32)]

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
# turn off GPU if converting int8
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

converter = tf.lite.TFLiteConverter.from_saved_model(f"weights/test2.pb")

# int8 - Convert using CPU (not working in GPU - document here: https://www.tensorflow.org/lite/performance/post_training_quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8

# float16 - Convert using CPU and GPU
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open("weights/test2.tflite", "wb").write(tflite_model)
```

Now you can be able to run test in mobile or test TFLite model using `test_tflite.py`. Good luck!
