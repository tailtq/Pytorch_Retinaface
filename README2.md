## Convert model from PyTorch to Tensorflow Lite

1. Load PyTorch and export ONNX model

```python
import torch

net = RetinaFace(cfg=cfg_mobilenet, phase='test')
net = load_model(net, "mobilenet0.25_Final.pth", cpu=False)

x = torch.randn(1, 3, 480, 480, requires_grad=True)
torch_out = net(x)
torch.onnx.export(net,  # model being run
                  x,  # model input (or a tuple for multiple inputs)
                  "test2.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                'output': {0: 'batch_size'}})
```


2. Benchmark ONNX and PyTorch model

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
net = load_model(net, "mobilenet0.25_Final.pth", False)
net = net.to(device)
net.eval()

ort_session = onnxruntime.InferenceSession("test2.onnx")

img = cv2.imread("./test-img2.jpeg", cv2.IMREAD_COLOR)
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


3. Convert ONNX to Tensorflow model

Pull docker image of tf-nightly: `docker pull tensorflow/tensorflow:devel-gpu`

Run container: `docker run -it --rm tensorflow/tensorflow:devel-gpu python`

Install tf-nightly:
```shell
# Some layers are missing when using Tensorflow 1.15 (TF - TFLite).
# Also by using version 1.15, TF model getting from ONNX is automatically frozen (but it's not the case for new layer in TF2).
# So by some suggests, I switched to tf-nightly with CUDA 11.0 and CuDNN 8.0.
# Don't know why the heck should I do it because we're using tf-nightly version @@.
# But leverage its environment (CUDA, CuDNN is enough already)
pip install tf-nightly onnx-tf

onnx-tf convert -i test2.onnx -o test2.pb
```


4. Convert Tensorflow to Tensorflow Lite

```python
import tensorflow as tf

converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model("test2.pb")
converter.allow_custom_ops = True

tflite_model = converter.convert()
open("test2.tflite", "wb").write(tflite_model)
```