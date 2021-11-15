"""
Code to load images and generate datasets.
"""

import random
import math

import numpy as np
from PIL import Image, ImageDraw

def _paste_ball_in_bg(bg_img, ball_img, params):
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

    shape = params.get("ground_truth_shape") or "rect"

    if shape == "rect":
        truth_draw.rectangle((x, y, x+w, y+h), fill=1)
    elif shape == "ellipse":
        truth_draw.ellipse((x, y, x+w, y+h), fill=1)

    return truth_img

def _gen_img_pair(bg_path, ball_path, params, evaluation):

    import tensorflow as tf

    with Image.open(ball_path) as ball_img:
        with Image.open(bg_path) as bg_img:
            ball_img = ball_img.convert("RGB")
            bg_img = bg_img.convert("RGB")

            if not evaluation and "train_val_img_size" in params:
                w, h = params["train_val_img_size"]
                bg_img = bg_img.crop((0, 0, w, h))

            truth_img = _paste_ball_in_bg(bg_img, ball_img, params)

            bg_array = _preprocess_img(tf.keras.preprocessing.image.img_to_array(bg_img))
            truth_array = tf.keras.preprocessing.image.img_to_array(truth_img)

            return bg_array, truth_array

def _preprocess_img(img_array):
    return img_array / np.max(img_array)

def image_generator(folder_path, ball_path, batch_size, num_batches=None,
        shuffle=True, params={}, evaluation=False):
    """
    Generator that loads images and creates dataset.

    Yields two numpy arrays, one is the input and the other is the ground truth.
    The shape of the arrays is [batch_size, height, width].

    It can be used directly with tf.data.Dataset.from_generator()

    params is a dict that can give information. For example to create smaller
    images, or create a different kind of ground truth.

    evaluation indicates if this generator used for training/validation or for
    testing evaluation. It is needed because some params are ignored in that
    case.
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
        max_shape = 0, 0 # Width and height of biggest image

        for i in range(batch_size):

            index = batch * batch_size + i
            x, y = _gen_img_pair(images_paths[index], ball_path, params, evaluation)
            x_imgs.append(x)
            y_imgs.append(y)

            max_shape = max(max_shape[0], x.shape[0]), max(max_shape[1], x.shape[1])

        # Pad images to maximum size plus what needed to make the size a
        # multiple of a specific number given in the params

        d = params["divisibility"]
        out_shape = (
                math.ceil(max_shape[0] / d) * d,
                math.ceil(max_shape[1] / d) * d
            )

        for i in range(batch_size):
            x_out = np.zeros((*out_shape, 3))
            y_out = np.zeros((*out_shape, 1))
            x_out[:x_imgs[i].shape[0], :x_imgs[i].shape[1], :] = x_imgs[i]
            y_out[:y_imgs[i].shape[0], :y_imgs[i].shape[1], :] = y_imgs[i]

            x_imgs[i] = x_out
            y_imgs[i] = y_out

        yield np.array(x_imgs), np.array(y_imgs)
