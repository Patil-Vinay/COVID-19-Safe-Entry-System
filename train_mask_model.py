"""Train the mask / no mask classifier on a PC and export it for the Pi.

Dataset folder layout (made by split_dataset.py):
    dataset/train/with_mask/*.jpg
    dataset/train/without_mask/*.jpg
    dataset/val/...  same two folders, used to decide when to stop
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 32

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="dataset")
parser.add_argument("--epochs", type=int, default=50, help="most runs stop earlier")
parser.add_argument("--output", default="mask_model")
args = parser.parse_args()

# folders are read in alphabetical order, so with_mask = 0 and without_mask = 1
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(args.dataset, "train"), seed=42,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode="binary")
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(args.dataset, "val"), shuffle=False,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode="binary")
print("Classes:", train_ds.class_names)

# augmentation is done on the dataset and not inside the model
# so the exported tflite model stays simple
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# MobileNetV2 trained on imagenet, only the new top layers are trained
base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                         include_top=False, weights="imagenet")
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs)  # 0-255 to -1..1
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # chance of no mask
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# stop when the val loss has not gone down by at least 0.001 for 5 epochs
# and go back to the best weights
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=5,
                                              restore_best_weights=True)
history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                    callbacks=[early_stop])
# weights are only put back if training stopped early, else the last epoch is kept
if early_stop.stopped_epoch > 0:
    best = early_stop.best_epoch
else:
    best = len(history.history["val_loss"]) - 1
print("Kept epoch %d of %d, val accuracy %.4f" % (best + 1, len(history.history["val_loss"]),
                                                 history.history["val_accuracy"][best]))

model.save(args.output + ".h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# float16 weights make the file half the size and it still runs fast,
# the default 8 bit version took over a second per image
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
with open(args.output + ".tflite", "wb") as f:
    f.write(converter.convert())
print("Saved", args.output + ".h5 and", args.output + ".tflite")

# loss and accuracy plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.axvline(best, color="gray", linestyle="--", label="kept")
plt.title("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.axvline(best, color="gray", linestyle="--", label="kept")
plt.title("Accuracy")
plt.legend()
os.makedirs("results", exist_ok=True)
plt.savefig("results/training_plot.png")
