#!/usr/bin/env python3

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

ball_img_path = Path("./res/ball.jpg")
val2017_folder_path = Path("/home/mbernardi/extra/async/ipcv/sem_3/deep_learning/labs/5/val2017")

def paste_ball_in_bg(bg_img, ball_img):
    """
    Pastes the ball into the background image in a random position.

    Modifies the bg_img, and also returns an extra image with the ground truth
    which has ones in the location of the pasted ball.

    Inputs and outputs are PIL Images
    """
    bg_w, bg_h = bg_img.size
    w, h = 50, 50 # Ball size in output image
    x = random.randint(0, bg_w - w)
    y = random.randint(0, bg_h - h)

    bg_img.paste(ball_img.resize((w, h)), (x, y))

    truth_img = Image.new(mode="1", size=(bg_w, bg_h), color=0)
    truth_draw = ImageDraw.Draw(truth_img)
    truth_draw.rectangle((x, y, x+w, y+h), fill=1)
    return truth_img

def image_generator(folder_path, ball_path, batch_size, num_batches=None, shuffle=True):

    def gen_img_pair(bg_path, ball_path):

        with Image.open(ball_path) as ball_img:
            with Image.open(bg_path) as bg_img:
                ball_img = ball_img.convert("RGB")
                bg_img = bg_img.convert("RGB")

                # TODO
                bg_img = bg_img.crop((0, 0, 200, 200))

                truth_img = paste_ball_in_bg(bg_img, ball_img)
                return bg_img, truth_img

    def preprocess_img(img_array):
        return img_array / np.max(img_array)

    def to_tensors(imgs):
        x_img, y_img = imgs
        return (
            tf.convert_to_tensor(preprocess_img(
                tf.keras.preprocessing.image.img_to_array(x_img)
            )),
            tf.convert_to_tensor(
                tf.keras.preprocessing.image.img_to_array(y_img)
            ),
        )


    images_paths = list(folder_path.glob("*.jpg"))
    if shuffle:
        random.shuffle(images_paths)
    num_images = len(images_paths)
    if not num_batches:
        num_batches = int(np.ceil(num_images/batch_size))

    for batch in range(num_batches):

        x_imgs = []
        y_imgs = []

        for i in range(batch_size):

            index = batch * batch_size + i
            x, y = to_tensors(gen_img_pair(images_paths[index], ball_path))
            x_imgs.append(x)
            y_imgs.append(y)

        yield np.array(x_imgs), np.array(y_imgs)

batch_size = 16
num_batches = 10

train_dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(val2017_folder_path, ball_img_path, batch_size, num_batches, shuffle=True),
        output_signature=(
            tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
        )
    )

val_dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(val2017_folder_path, ball_img_path, batch_size, num_batches, shuffle=False),
        output_signature=(
            tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
        )
    )


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))#, input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=2,
    activation='sigmoid', padding='same'))
model.add(tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=2,
    activation='sigmoid', padding='same'))

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy')

model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        verbose=1,
    )

image_gen = image_generator(val2017_folder_path, ball_img_path, batch_size=16, shuffle=False)
x, y = next(image_gen)
img = x[0]
y_est = model.predict(x)
plt.imshow(y_est[0].squeeze())
plt.show()
