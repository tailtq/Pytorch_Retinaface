import cv2
from image2face import RetinafaceDetection

if __name__ == "__main__":
    face_detection = RetinafaceDetection("mobile0.25", use_cpu=True)

    img = cv2.imread("test.jpg", cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(face_detection.predict(img).shape)
