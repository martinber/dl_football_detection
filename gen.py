"""
Code to load images and generate datasets.
"""

import random

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

def _paste_ball_in_bg(bg_img, ball_img):
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

def _gen_img_pair(bg_path, ball_path):

    with Image.open(ball_path) as ball_img:
        with Image.open(bg_path) as bg_img:
            ball_img = ball_img.convert("RGB")
            bg_img = bg_img.convert("RGB")

            # TODO
            bg_img = bg_img.crop((0, 0, 200, 200))

            truth_img = _paste_ball_in_bg(bg_img, ball_img)
            return bg_img, truth_img

def _preprocess_img(img_array):
    return img_array / np.max(img_array)

def _to_tensors(imgs):
    x_img, y_img = imgs
    return (
        tf.convert_to_tensor(_preprocess_img(
            tf.keras.preprocessing.image.img_to_array(x_img)
        )),
        tf.convert_to_tensor(
            tf.keras.preprocessing.image.img_to_array(y_img)
        ),
    )

def image_generator(folder_path, ball_path, batch_size, num_batches=None, shuffle=True):
    """
    Generator that loads images and creates dataset.

    Yields two numpy arrays, one is the input and the other is the ground truth.
    The shape of the arrays is [batch_size, height, width].

    It can be used directly with tf.data.Dataset.from_generator()
    """

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
            x, y = _to_tensors(_gen_img_pair(images_paths[index], ball_path))
            x_imgs.append(x)
            y_imgs.append(y)

        print(np.array(x_imgs).shape)

        yield np.array(x_imgs), np.array(y_imgs)
