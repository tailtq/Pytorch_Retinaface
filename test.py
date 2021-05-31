import cv2
import numpy as np
from image2face import RetinafacePrediction, ArcfacePrediction

if __name__ == "__main__":
    face_detection = RetinafacePrediction("resnet50", use_cpu=True)
    img = cv2.imread("Thai.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detections = face_detection.predict(img)
    print("Detection shape:", detections.shape)

    for detection in detections:
        detection = detection.astype(int)
        cv2.rectangle(img, (detection[0], detection[1]), (detection[2], detection[3]), (0, 255, 0), 3)
        aligned_face = face_detection.align_face(img, detection)

        cv2.imshow("Test", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imshow("Test1", cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
        cv2.waitKey(-1)

    face_recognition = ArcfacePrediction("resnet50", use_cpu=True)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    print("Recognition shape:", face_recognition.predict(img).shape)
