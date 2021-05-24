import cv2
import torch


class BasePrediction:
    def __init__(self, use_cpu=False):
        self.use_cpu = use_cpu

    def predict(self, img, width=480):
        pass

    def predict_batch(self, imgs: list, width=480):
        pass

    @staticmethod
    def resize_img(img, width):
        origin_h, origin_w, _ = img.shape

        return cv2.resize(img, (width, int(origin_h * width / origin_w)))

    def _get_device(self):
        return torch.device("cpu" if self.use_cpu else "cuda:0")
