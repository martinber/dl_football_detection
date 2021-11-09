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

def random_path_generator(folder_path):
    """
    Generator that returns random file paths in the given folder.

    Only files ending with .jpg
    """
    files = list(folder_path.glob("*.jpg"))
    while True:
        yield random.choice(files)

def image_generator(path_gen, ball_img_path):
    """
    Get a tuple with a generated image and ground truth
    """

    with Image.open(ball_img_path) as ball_img:
        while True:
            bg_path = next(path_gen)
            with Image.open(bg_path) as bg_img:
                truth_img = paste_ball_in_bg(bg_img, ball_img)

                yield bg_img, truth_img

def batch_generator(image_gen):
    """
    Converts images to tensors and..
    """
    for bg_img, truth_img in image_gen:
        bg_tensor = tf.convert_to_tensor(tf.keras.preprocessing.image.img_to_array(bg_img))
        truth_tensor = tf.convert_to_tensor(tf.keras.preprocessing.image.img_to_array(bg_img))

        yield bg_tensor, truth_tensor


path_gen = random_path_generator(val2017_folder_path)
image_gen = image_generator(path_gen, ball_img_path)
batch_gen = batch_generator(image_gen)



x, y = next(image_gen)
x.show()
y.show()


next(batch_gen)


dataset = tf.data.Dataset.from_generator(
        lambda: batch_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
        )
    )

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))

