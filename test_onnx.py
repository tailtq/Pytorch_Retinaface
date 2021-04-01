import cv2
import torch
import onnxruntime
import numpy as np
import time

from data import cfg_re50, cfg_mnet
from detect import load_model
from models.retinaface import RetinaFace


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


device = torch.device("cuda")

torch.set_grad_enabled(False)
net = RetinaFace(cfg=cfg_mnet, phase='test')
net = load_model(net, "weights/mobilenet0.25_Final.pth", False)
net = net.to(device)
net.eval()

ort_session = onnxruntime.InferenceSession("weights/test2.onnx")

img = cv2.imread("imgs/test-img2.jpeg", cv2.IMREAD_COLOR)
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
test = ort_session.run(None, ort_inputs)
loc2, conf2, landms2 = test

print(test[0].shape, test[1].shape, test[2].shape)
print('==================================')
print('loc', loc.shape)
print('loc2', loc2.shape)

print('==================================')
print('conf', conf.shape)
print('conf2', conf2.shape)

print('==================================')
print('landms', landms.shape)
print('landms2', landms2.shape)
# print(net)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(loc), loc2, rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(to_numpy(conf), conf2, rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(to_numpy(landms), landms2, rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
