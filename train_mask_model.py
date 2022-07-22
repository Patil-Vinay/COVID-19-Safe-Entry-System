"""Train the mask / no mask classifier on a PC and export it for the Pi.

Dataset folder layout:
    dataset/with_mask/*.jpg
    dataset/without_mask/*.jpg
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 32

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="dataset")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--output", default="mask_model")
args = parser.parse_args()

# folders are read in alphabetical order, so with_mask = 0 and without_mask = 1
train_ds = tf.keras.utils.image_dataset_from_directory(
    args.dataset, validation_split=0.2, subset="training", seed=42,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode="binary")
val_ds = tf.keras.utils.image_dataset_from_directory(
    args.dataset, validation_split=0.2, subset="validation", seed=42,
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

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

model.save(args.output + ".h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
with open(args.output + ".tflite", "wb") as f:
    f.write(converter.convert())
print("Saved", args.output + ".h5 and", args.output + ".tflite")

# loss and accuracy plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()
plt.savefig("training_plot.png")
