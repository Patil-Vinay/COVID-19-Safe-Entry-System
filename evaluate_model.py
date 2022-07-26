# Checks the tflite model on the test images. This is the same model file
# that runs on the Pi, so the numbers include the size reduction from converting.
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from mask_detector import Interpreter

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="dataset/test")
parser.add_argument("--model", default="mask_model.tflite")
parser.add_argument("--limit", type=float, default=0.5, help="score above this is no mask")
args = parser.parse_args()

interpreter = Interpreter(model_path=args.model)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
size = inp["shape"][1]

classes = ["with_mask", "without_mask"]
y_true, y_pred = [], []
for label, cls in enumerate(classes):
    folder = os.path.join(args.dataset, cls)
    for name in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, name))
        if img is None:
            print("Could not read", name)
            continue
        # same steps as MaskDetector.no_mask_score
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        interpreter.set_tensor(inp["index"], np.expand_dims(img.astype(np.float32), axis=0))
        interpreter.invoke()
        score = interpreter.get_tensor(out["index"])[0][0]
        y_true.append(label)
        y_pred.append(1 if score > args.limit else 0)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
print("Test images:", len(y_true))
print("Accuracy: %.2f%%" % (100 * np.mean(y_true == y_pred)))
print(classification_report(y_true, y_pred, target_names=classes, digits=3))

cm = confusion_matrix(y_true, y_pred)
print(cm)
os.makedirs("results", exist_ok=True)
ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues")
plt.title("Test set")
plt.savefig("results/confusion_matrix.png", bbox_inches="tight")
print("Saved results/confusion_matrix.png")
