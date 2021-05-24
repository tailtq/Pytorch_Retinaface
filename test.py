import cv2
import numpy as np
from image2face import RetinafacePrediction, ArcfacePrediction

if __name__ == "__main__":
    face_detection = RetinafacePrediction("resnet50", use_cpu=True)
    img = cv2.imread("test.jpg", cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(face_detection.predict(img).shape)

    face_recognition = ArcfacePrediction("resnet50", use_cpu=True)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)

    print(face_recognition.predict(img).shape)
