import os

import cv2
import numpy as np

# tflite_runtime on the Pi, full tensorflow works too on a PC
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

HERE = os.path.dirname(os.path.abspath(__file__))


class MaskDetector:
    def __init__(self, model_path=os.path.join(HERE, "mask_model.tflite"),
                 prototxt=os.path.join(HERE, "face_detector", "deploy.prototxt"),
                 weights=os.path.join(HERE, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel"),
                 face_confidence=0.5):
        for path in (model_path, prototxt, weights):
            if not os.path.exists(path):
                raise FileNotFoundError(path + " not found, see the README")
        # OpenCV's SSD face detector, works better on masked faces than haar cascades
        self.face_net = cv2.dnn.readNetFromCaffe(prototxt, weights)
        self.face_confidence = face_confidence

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.size = self.input["shape"][1]

    def find_face(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        # keep the biggest face, that is the person standing at the camera
        best = None
        best_area = 0
        for i in range(detections.shape[2]):
            if detections[0, 0, i, 2] < self.face_confidence:
                continue
            x1, y1, x2, y2 = [int(v) for v in detections[0, 0, i, 3:7] * np.array([w, h, w, h])]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            area = (x2 - x1) * (y2 - y1)
            if x2 > x1 and y2 > y1 and area > best_area:
                best = (x1, y1, x2, y2)
                best_area = area
        return best

    def no_mask_score(self, face):
        # model takes RGB 0-255 and gives the chance of no mask
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (self.size, self.size))
        face = np.expand_dims(face.astype(np.float32), axis=0)
        self.interpreter.set_tensor(self.input["index"], face)
        self.interpreter.invoke()
        return float(self.interpreter.get_tensor(self.output["index"])[0][0])

    def check(self, frame):
        """Returns (box, no_mask_score), both None if no face is found."""
        box = self.find_face(frame)
        if box is None:
            return None, None
        x1, y1, x2, y2 = box
        return box, self.no_mask_score(frame[y1:y2, x1:x2])


# quick test with the webcam
if __name__ == "__main__":
    detector = MaskDetector()
    cam = cv2.VideoCapture(0)
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        box, score = detector.check(frame)
        if box is not None:
            label = "No Mask" if score > 0.5 else "Mask"
            color = (0, 0, 255) if score > 0.5 else (0, 255, 0)
            cv2.rectangle(frame, box[:2], box[2:], color, 2)
            cv2.putText(frame, "%s %.2f" % (label, score), (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Mask check", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()
