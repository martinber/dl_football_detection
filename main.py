#!/usr/bin/env python3

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import tensorflow as tf

import gen

BALL_IMG_PATH = Path("./res/ball.jpg")
VAL2017_FOLDER_PATH = Path("/home/mbernardi/extra/async/ipcv/sem_3/deep_learning/labs/5/val2017")

def define_case():
    """
    Define and compile the model and all hyperparameters for one case
    """

    # Dictionary with model and hyperparameters
    case = {
        # Samples per batch
        "batch_size": 16,

        # Number of batches per epoch
        "num_batches": 3,

        # Number of epochs
        "num_epochs": 10,

        # Optimizer and loss
        "optimizer": tf.keras.optimizers.Adam(1e-3),
        "loss": 'binary_crossentropy',

        # Model
        "model": tf.keras.Sequential(),
    }

    case["model"].add(tf.keras.layers.Conv2D(
            32, (3, 3),
            activation='relu', padding='same',
            #, input_shape=(32, 32, 3)))
        ))
    case["model"].add(tf.keras.layers.MaxPooling2D((2, 2)))
    case["model"].add(tf.keras.layers.Conv2D(
            64, (3, 3),
            activation='relu', padding='same',
        ))
    case["model"].add(tf.keras.layers.MaxPooling2D((2, 2)))
    # case["model"].add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    # case["model"].add(tf.keras.layers.MaxPooling2D((2, 2)))
    case["model"].add(tf.keras.layers.Conv2DTranspose(
            16, (3, 3), strides=2,
            activation='sigmoid', padding='same',
        ))
    case["model"].add(tf.keras.layers.Conv2DTranspose(
            1, (3, 3), strides=2,
            activation='sigmoid', padding='same',
        ))

    case["model"].compile(optimizer=case["optimizer"], loss=case["loss"])

    return case

def main():

    case = define_case()

    train_dataset = tf.data.Dataset.from_generator(
            lambda: gen.image_generator(
                VAL2017_FOLDER_PATH,
                BALL_IMG_PATH,
                batch_size=case["batch_size"],
                num_batches=case["num_batches"],
                shuffle=True,
            ),
            output_signature=(
                tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
            )
        )

    val_dataset = tf.data.Dataset.from_generator(
            lambda: gen.image_generator(
                VAL2017_FOLDER_PATH,
                BALL_IMG_PATH,
                batch_size=case["batch_size"],
                num_batches=case["num_batches"],
                shuffle=False,
            ),
            output_signature=(
                tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
            )
        )

    case["model"].fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=case["num_epochs"],
            verbose=1,
        )

    image_gen = gen.image_generator(VAL2017_FOLDER_PATH, BALL_IMG_PATH, batch_size=16, shuffle=False)
    x, y = next(image_gen)
    img = x[0]
    y_est = case["model"].predict(x)
    plt.imshow(y_est[0].squeeze())
    plt.show()

if __name__ == "__main__":
    main()
