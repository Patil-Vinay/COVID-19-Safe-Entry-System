# Splits the images into dataset/train, dataset/val and dataset/test.
# The dataset has many copies of the same photo (saved twice, mirrored, edited
# or with a mask pasted on), so similar images are put in a group and a whole
# group always goes to the same folder. Otherwise the model would already have
# seen most of the test photos and the test accuracy would be too high.
import argparse
import hashlib
import os
import random
import shutil

import cv2
import numpy as np


def dhash(img):
    # 256 bit difference hash, small changes in the image only flip a few bits
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (17, 16))
    return (g[:, 1:] > g[:, :-1]).flatten()


parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True, help="folder with with_mask and without_mask")
parser.add_argument("--dest", default="dataset")
parser.add_argument("--test", type=float, default=0.2, help="part of images kept for testing")
parser.add_argument("--val", type=float, default=0.1, help="part of images used for validation")
parser.add_argument("--max-diff", type=int, default=44, help="hash bits (of 256) that can differ for two similar images")
args = parser.parse_args()

classes = ("with_mask", "without_mask")
files = []
whole, mirror, top, top_mirror = [], [], [], []
seen = set()
skipped = 0
for cls in classes:
    for name in sorted(os.listdir(os.path.join(args.source, cls))):
        path = os.path.join(args.source, cls, name)
        with open(path, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        img = cv2.imread(path)
        if md5 in seen or img is None:
            skipped += 1
            continue
        seen.add(md5)
        files.append((cls, name))
        upper = img[: img.shape[0] // 2]  # eyes and hair, not covered by a mask
        whole.append(dhash(img))
        mirror.append(dhash(cv2.flip(img, 1)))
        top.append(dhash(upper))
        top_mirror.append(dhash(cv2.flip(upper, 1)))
print(len(files), "images,", skipped, "exact copies or unreadable files skipped")

whole, mirror = np.array(whole), np.array(mirror)
top, top_mirror = np.array(top), np.array(top_mirror)

# join similar images into groups (union find)
parent = list(range(len(files)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for i in range(len(files)):
    d_whole = np.minimum((whole != whole[i]).sum(1), (mirror != whole[i]).sum(1))
    d_top = np.minimum((top != top[i]).sum(1), (top_mirror != top[i]).sum(1))
    for j in np.where((d_whole <= args.max_diff) | (d_top <= args.max_diff))[0]:
        parent[find(j)] = find(i)

groups = {}
for i in range(len(files)):
    groups.setdefault(find(i), []).append(i)
groups = list(groups.values())
print(len(groups), "groups of similar photos, biggest has", max(len(g) for g in groups), "images")

# fill test first, then val, the rest is train
random.seed(42)
random.shuffle(groups)
n_test = args.test * len(files)
n_val = args.val * len(files)
counts = {part: {cls: 0 for cls in classes} for part in ("train", "val", "test")}
used = 0
for g in groups:
    if used < n_test:
        part = "test"
    elif used < n_test + n_val:
        part = "val"
    else:
        part = "train"
    used += len(g)
    for i in g:
        cls, name = files[i]
        out = os.path.join(args.dest, part, cls)
        os.makedirs(out, exist_ok=True)
        shutil.copy(os.path.join(args.source, cls, name), out)
        counts[part][cls] += 1

for part in ("train", "val", "test"):
    print(part, counts[part])
