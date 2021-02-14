# Face detection using MTCNN
This is the implementation of Multi task Cascaded Convolutional Neural Network for face detection by *Zhang, K et al. (2016)* [ZHANG2016](https://arxiv.org/pdf/1604.02878)
The code has been forked from the official implementation of MTCNN by *ipazc* [MTCNN](https://github.com/ipazc/mtcnn)

## Installation
Use the requirements.txt file to install the dependencies.

`pip3 install -r requirements.txt`

## Usage
Use the jupyter notebook [face detection.ipynb](https://github.com/Aasish4/Face-detection-using-MTCNN/blob/main/Face%20detection.ipynb) for testing the implementation of MTCNN. All codes are beautifully quoted for knowledge at every steps.

## Output
This is the output of the model with bounding box detection and keypoints.\
![output](https://github.com/Aasish4/Face-detection-using-MTCNN/blob/main/output.jpg)

## code snippet
```javascript
>>> from mtcnn.mtcnn import MTCNN
>>> import cv2

>>> img = cv2.cvtColor(cv2.imread('image.jpg'), cv2.COLOR_BGR2RGB)
>>> detector = MTCNN()
>>> detector.detect_faces(img)
[
    {
        'box': [277, 90, 48, 63],
        'keypoints':
        {
            'nose': (303, 131),
            'mouth_right': (313, 141),
            'right_eye': (314, 114),
            'left_eye': (291, 117),
            'mouth_left': (296, 143)
        },
        'confidence': 0.99851983785629272
    }
]
```
