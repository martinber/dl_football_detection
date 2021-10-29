#!/usr/bin/env python3

from pathlib import Path
import random

from PIL import Image, ImageDraw
import tensorflow as tf

ball_img_path = Path("./res/ball.jpg")
val2017_folder_path = Path("/home/mbernardi/extra/async/ipcv/sem_3/deep_learning/labs/5/val2017")

def paste_ball_in_bg(bg_img, ball_img):
    """
    Pastes the ball into the background image in a random position.

    Modifies the bg_img, and also returns an extra image with the ground truth
    which has ones in the location of the pasted ball.
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

def random_path_generator(folder_path):
    files = list(folder_path.glob("*.jpg"))
    while True:
        yield random.choice(files)

def data_generator(ball_img_path, val2017_folder_path):
    """
    Get a tuple with a generated image and ground truth
    """

    bg_paths = random_path_generator(val2017_folder_path)

    with Image.open(ball_img_path) as ball_img:
        while True:
            bg_path = next(bg_paths)
            with Image.open(bg_path) as bg_img:
                truth_img = paste_ball_in_bg(bg_img, ball_img)

                yield bg_img, truth_img


gen = data_generator(ball_img_path, val2017_folder_path)
x, y = next(gen)
x.show()
y.show()

# dataset = tf.data.Dataset.from_generator(
#         lambda: data_generator(ball_img_path, val2017_folder_path),
#         (tf.float32, tf.int16))

