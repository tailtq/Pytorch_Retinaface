from __future__ import print_function
import os
import cv2
import torch
import numpy as np
from pathlib import Path

from ..base_prediction import BasePrediction
from .config import cfg_mobilenet, cfg_re50
from .layers.functions.prior_box import PriorBox
from .utils.nms.py_cpu_nms import py_cpu_nms
from .models.retinaface import RetinaFace
from .utils import decode, decode_landm, download_file_from_drive


dir_path = Path(__file__).parent
torch.set_grad_enabled(False)


class RetinafacePrediction(BasePrediction):
    network_cfgs = {
        "mobile0.25": cfg_mobilenet,
        "resnet50": cfg_re50,
    }
    backbone_paths = {
        "mobile0.25": dir_path / "weights/mobilenet0.25_Final.pth",
        "resnet50": dir_path / "weights/Retinaface_Resnet50_Final.pth",
    }

    @torch.no_grad()
    def __init__(self, backbone, confidence_threshold=0.5, nms_threshold=0.4, use_cpu=False):
        super().__init__(use_cpu)

        self.backbone = backbone
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.use_cpu = use_cpu
        self.cfg = self.network_cfgs[self.backbone]

        self.model = self._load_trained_model(self.cfg, self.backbone_paths[self.backbone], self.use_cpu)

    def predict(self, img, width=480):
        """
            Output shape of the function is a matrix having shape (x, 15)
            - 0-3: top left and bottom right coordinates of face bounding box
            - 4: confidence
            - 5-14: landmarks 
        """
        device = self._get_device()
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
        loc, conf, landmarks = self.model(img)

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
        landmarks = decode_landm(landmarks.data.squeeze(0), prior_data, self.cfg['variance'])
        landmarks = landmarks * scale_landmark
        landmarks = landmarks.cpu().numpy()
        landmarks = landmarks[indices]

        # do nms
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landmarks = landmarks[keep]

        dets = np.concatenate((dets, landmarks), axis=1)

        return dets

    @staticmethod
    def align_face(image, detection):
        x1, y1, x2, y2 = detection[:4].astype(int)
        face = image[y1:y2, x1:x2]

        landmarks = detection[5:]
        landmarks[::2] -= x1
        landmarks[1::2] -= y1

        left_eye_x, left_eye_y, right_eye_x, right_eye_y = landmarks[:4]
        dx = right_eye_x - left_eye_x
        dy = right_eye_y - left_eye_y
        # calculate the angle between x-axis and the vector of 2 eyes
        angle = np.degrees(np.arctan2(dy, dx))

        # calculate the center of eyes
        center_eyes_x = int((left_eye_x + right_eye_x) / 2)
        center_eyes_y = int((left_eye_y + right_eye_y) / 2)

        # create rotation matrix base on the center and angle without scaling
        rotation_matrix = cv2.getRotationMatrix2D((center_eyes_x, center_eyes_y), angle, scale=1)
        # align face
        aligned_face = cv2.warpAffine(face, rotation_matrix, (face.shape[1], face.shape[0]), flags=cv2.INTER_CUBIC)

        return aligned_face

    # def align_face(self, image, landmarks, desiredLeftEye=(0.35, 0.35), desiredFaceWidth=256):
    #     left_eye_x, left_eye_y, right_eye_x, right_eye_y = landmarks[:4]
    #     dx = right_eye_x - left_eye_x
    #     dy = right_eye_y - left_eye_y
    #
    #     desiredFaceHeight = desiredFaceWidth
    #     desiredRightEyeX = 1.0 - desiredLeftEye[0]
    #     # determine the scale of the new resulting image by taking
    #     # the ratio of the distance between eyes in the *current*
    #     # image to the ratio of distance between eyes in the
    #     # *desired* image
    #     dist = np.sqrt((dx ** 2) + (dy ** 2)) * 1.1
    #     desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    #     desiredDist *= desiredFaceWidth
    #     scale = desiredDist / dist
    #
    #     tX = desiredFaceWidth * 0.5
    #     tY = desiredFaceHeight * desiredLeftEye[1]
    #     # rotation_matrix[0, 2] += (tX - eyesCenter[0])
    #     # rotation_matrix[1, 2] += (tY - eyesCenter[1])
    #
    #
    #     angle = np.degrees(np.arctan2(dy, dx))
    #     center_eyes_x = int((left_eye_x + right_eye_x) / 2)
    #     center_eyes_y = int((left_eye_y + right_eye_y) / 2)
    #
    #     rotation_matrix = cv2.getRotationMatrix2D((center_eyes_x, center_eyes_y), angle, scale)
    #     # rotation_matrix = cv2.getRotationMatrix2D((center_eyes_x, center_eyes_y), angle, 1)
    #     rotation_matrix[0, 2] += tX - center_eyes_x
    #     rotation_matrix[1, 2] += tY - center_eyes_y
    #
    #     aligned_face = cv2.warpAffine(image, rotation_matrix, (desiredFaceWidth, desiredFaceHeight), flags=cv2.INTER_CUBIC)
    #
    #     return aligned_face

    def _load_trained_model(self, cfg, trained_path, use_cpu):
        if not os.path.exists(trained_path):
            print("Download model...")
            download_file_from_drive("1jLd-yASoo34hqrG0F0k9NV04SYCPD4Qa", trained_path)

        model = RetinaFace(cfg=cfg, phase="test")
        device = self._get_device()

        print("Load Retinaface model successfully")

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
    def _check_keys(model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys

        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'

        return True

    @staticmethod
    def _remove_prefix(state_dict, prefix):
        """ Old style model is stored with all names of parameters sharing common prefix 'module.' """
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x

        return {f(key): value for key, value in state_dict.items()}
