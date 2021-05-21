from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "Face Recognition package"
LONG_DESCRIPTION = "Face Recognition package"

# reference: https://www.freecodecamp.org/news/build-your-first-python-package/
setup(
    name="image2face",
    version=VERSION,
    author="Tai Le",
    author_email="<ltquoctaidn98@gmail.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch==1.7.0",
        "torchvision",
        "opencv-python==4.4.0"
    ],
    keywords=["python", "computer vision", "face detection"],
    classifiers=[
        "Operating System :: OS Independent",
    ]
)
