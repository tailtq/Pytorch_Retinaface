from __future__ import print_function
import os
import cv2
import torch
import numpy as np
from pathlib import Path

from config import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils import decode, decode_landm, download_file_from_drive


dir_path = Path(__file__).parent

network_cfgs = {
    "mobile0.25": cfg_mnet,
    "resnet50": cfg_re50,
}
model_paths = {
    "mobile0.25": dir_path / "weights/mobilenet0.25_Final.pth",
    "resnet50": dir_path / "weights/Retinaface_Resnet50_Final.pth",
}
torch.set_grad_enabled(False)


class RetinafaceDetection:
    def __init__(self, network_name, confidence_threshold=0.5, nms_threshold=0.4, use_cpu=False):
        self.network_name = network_name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.use_cpu = use_cpu
        self.cfg = network_cfgs[self.network_name]

        self.model = self._load_trained_model(self.cfg, model_paths[self.network_name], self.use_cpu)

    def predict(self, img, width=480):
        device = self._get_device(self.use_cpu)
        origin_h, origin_w, _ = img.shape

        # resize image to reduce inference time
        img = cv2.resize(img, (width, int(origin_h * width / origin_w)))
        new_h, new_w, _ = img.shape
        img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)
        img = img.to(device)

        # define scale ratio
        scale_box = torch.Tensor([origin_w, origin_h, origin_w, origin_h])
        scale_box = scale_box.to(device)

        scale_landmark = torch.Tensor([origin_w, origin_h, origin_w, origin_h,
                                       origin_w, origin_h, origin_w, origin_h,
                                       origin_w, origin_h])
        scale_landmark = scale_landmark.to(device)
        # predict bounding boxes + landmarks
        loc, conf, landms = self.model(img)

        # get objects having confidence greater or equal to threshold
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        indices = np.where(scores >= self.confidence_threshold)
        scores = scores[indices]

        # initialize anchor box generator
        priorbox = PriorBox(self.cfg, image_size=(new_h, new_w))
        priors = priorbox.forward()
        priors = priors.to(device)
        # generate anchor boxes
        prior_data = priors.data
        # decode bounding boxes by anchor boxes and model prediction's location
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale_box
        boxes = boxes.cpu().numpy()
        boxes = boxes[indices]

        # decode bounding boxes by anchor boxes and model prediction's location
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        landms = landms * scale_landmark
        landms = landms.cpu().numpy()
        landms = landms[indices]

        # do nms
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # tl_x, tl_y, br_x, br_y, conf, landmark_1_x, landmark_1_y, ..., landmark_5_x, landmark_5_y
        dets = np.concatenate((dets, landms), axis=1)

        return dets

    def predict_batch(self, imgs: list):
        return

    def _load_trained_model(self, cfg, trained_path, use_cpu):
        if not os.path.exists(trained_path):
            print("Download model...")
            download_file_from_drive("1jLd-yASoo34hqrG0F0k9NV04SYCPD4Qa", trained_path)

        print("Loading pretrained model from {}".format(trained_path))
        model = RetinaFace(cfg=cfg, phase="test")
        device = self._get_device(use_cpu)

        if use_cpu:
            pretrained_dict = torch.load(trained_path, map_location=lambda storage, loc: storage)
        else:
            pretrained_dict = torch.load(trained_path, map_location=lambda storage, loc: storage.cuda(device))

        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self._remove_prefix(pretrained_dict["state_dict"], "module.")
        else:
            pretrained_dict = self._remove_prefix(pretrained_dict, "module.")

        self._check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        model.eval()
        model = model.to(device)

        return model

    @staticmethod
    def _get_device(use_cpu):
        return torch.device("cpu" if use_cpu else "cuda:0")

    @staticmethod
    def _check_keys(model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys

        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'

        return True

    @staticmethod
    def _remove_prefix(state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x

        return {f(key): value for key, value in state_dict.items()}


# if __name__ == "__main__":
#     face_detection = RetinafaceDetection("mobile0.25", use_cpu=True)
#
#     img = cv2.imread("test.jpg", cv2.IMREAD_COLOR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     print(face_detection.predict(img).shape)
