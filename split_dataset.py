# Copies the images into dataset/train and dataset/test for each class,
# so the test images are never seen during training.
# Some datasets have the same image saved twice, those copies are skipped.
import argparse
import hashlib
import os
import random
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True, help="folder with with_mask and without_mask")
parser.add_argument("--dest", default="dataset")
parser.add_argument("--test", type=float, default=0.2, help="part of images kept for testing")
args = parser.parse_args()

random.seed(42)
seen = set()
for cls in ("with_mask", "without_mask"):
    files = []
    skipped = 0
    for name in sorted(os.listdir(os.path.join(args.source, cls))):
        with open(os.path.join(args.source, cls, name), "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        if h in seen:
            skipped += 1
            continue
        seen.add(h)
        files.append(name)

    random.shuffle(files)
    n_test = int(len(files) * args.test)
    for i, name in enumerate(files):
        part = "test" if i < n_test else "train"
        out = os.path.join(args.dest, part, cls)
        os.makedirs(out, exist_ok=True)
        shutil.copy(os.path.join(args.source, cls, name), out)
    print(cls, len(files) - n_test, "train", n_test, "test", skipped, "duplicates skipped")
