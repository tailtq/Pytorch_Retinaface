from pathlib import Path
import numpy as np
import torch
import cv2

from .backbones import get_model
from ..base_prediction import BasePrediction

dir_path = Path(__file__).parent
torch.set_grad_enabled(False)


class ArcfacePrediction(BasePrediction):
    backbone_paths = {
        "resnet50": dir_path / "weights/resnet50.pth"
    }

    @torch.no_grad()
    def __init__(self, backbone, use_cpu=True):
        super().__init__(use_cpu)

        self.model = self._load_pretrained_model(backbone, self.backbone_paths[backbone])

    def predict(self, img, width=112):
        img = cv2.resize(img, (width, width))
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float() / 255
        img.div_(255).sub_(0.5).div_(0.5)

        img = img.to(self._get_device())

        features = self.model(img).numpy().squeeze(0)

        return features

    def _load_pretrained_model(self, backbone, trained_path):
        model = get_model(backbone, fp16=False)
        weight = torch.load(trained_path, map_location=self._get_device())
        model.load_state_dict(weight, strict=False)
        model.eval()

        return model

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
#     parser.add_argument('--network', type=str, default='r50', help='backbone network')
#     parser.add_argument('--weight', type=str, default='')
#     parser.add_argument('--img', type=str, default=None)
#     args = parser.parse_args()
#     inference(args.weight, args.network, args.img)
