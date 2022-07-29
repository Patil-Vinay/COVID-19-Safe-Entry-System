# Downloads the weights of OpenCV's face detector (about 10 MB) into face_detector/
# They are not kept in the repo because of the size.
import os
import urllib.request

URL = ("https://github.com/opencv/opencv_3rdparty/raw/"
       "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_detector",
                    "res10_300x300_ssd_iter_140000.caffemodel")

if os.path.exists(PATH):
    print(PATH, "is already there")
else:
    print("Downloading", URL)
    urllib.request.urlretrieve(URL, PATH)
    print("Saved", PATH, os.path.getsize(PATH) // 1024, "KB")
