import cv2
import numpy as np
from image2face import RetinafacePrediction, ArcfacePrediction

if __name__ == "__main__":
    face_detection = RetinafacePrediction("resnet50", use_cpu=True)
    img = cv2.imread("test3.jpg", cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detections = face_detection.predict(img)
    print("Detection shape:", detections.shape)

    for detection in detections:
        aligned_face = face_detection.align_face(img, detection)

    face_recognition = ArcfacePrediction("resnet50", use_cpu=True)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    print("Recognition shape:", face_recognition.predict(img).shape)
